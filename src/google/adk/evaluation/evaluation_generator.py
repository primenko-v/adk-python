# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import logging
import time
from typing import Any
from typing import AsyncGenerator
from typing import Optional
import uuid

from google.genai import errors
from google.genai import types
from google.genai.types import Content
import opentelemetry.context as context_api
from opentelemetry.trace import set_span_in_context
from opentelemetry.trace import Span
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosed
from websockets.exceptions import ConnectionClosedOK

from ..agents.callback_context import CallbackContext
from ..agents.live_request_queue import LiveRequestQueue
from ..agents.llm_agent import Agent
from ..agents.run_config import RunConfig
from ..agents.run_config import StreamingMode
from ..artifacts.base_artifact_service import BaseArtifactService
from ..artifacts.in_memory_artifact_service import InMemoryArtifactService
from ..events.event import Event
from ..memory.base_memory_service import BaseMemoryService
from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..models.llm_request import LlmRequest
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.in_memory_session_service import InMemorySessionService
from ..sessions.session import Session
from ..telemetry import tracing as _telemetry
from ..utils.context_utils import Aclosing
from ._retry_options_utils import EnsureRetryOptionsPlugin
from .app_details import AgentDetails
from .app_details import AppDetails
from .constants import DEFAULT_LIVE_TIMEOUT_SECONDS
from .constants import TRANSCRIPTION_CHUNK_METADATA_KEY
from .eval_case import EvalCase
from .eval_case import Invocation
from .eval_case import InvocationEvent
from .eval_case import InvocationEvents
from .eval_case import SessionInput
from .eval_set import EvalSet
from .request_intercepter_plugin import _RequestIntercepterPlugin
from .simulation.user_simulator import BaseUserSimulatorConfig
from .simulation.user_simulator import Status as UserSimulatorStatus
from .simulation.user_simulator import UserSimulator
from .simulation.user_simulator_provider import UserSimulatorProvider

logger = logging.getLogger("google_adk." + __name__)

_USER_AUTHOR = "user"
_DEFAULT_AUTHOR = "agent"

# Idle window for draining a turn's events. `turn_complete` only marks the
# end of generation; audio and its (server-side, decoupled) transcription can
# keep arriving for seconds afterwards. Any received event resets the window,
# so it only needs to outlast the gaps between events, not the whole tail.
_TRANSCRIPTION_TAIL_GRACE_SECONDS = 2.0


def _extract_content_text(content: Optional[Content]) -> Optional[str]:
  """Joins the text parts of a `Content` into a single string, or None."""
  if not content or not content.parts:
    return None
  text = " ".join(p.text for p in content.parts if p.text)
  return text or None


def _record_live_turn_telemetry(
    span: Span,
    events: list[Event],
    invocation_id: str,
    user_message: Optional[Content],
) -> None:
  """Records the turn's request/response and chronology on a `live_turn` span.

  The model may speak both before and after a tool call within one turn, so
  transcription chunks are stitched into one utterance per such segment: the
  `llm_response` attribute then mirrors the real "speech → tool → speech"
  order instead of fusing everything into one string. Each utterance, tool
  call and tool response is also recorded as a timestamped span event, giving
  the span a chronological view of the turn.
  """
  user_text = _extract_content_text(user_message)
  if user_text:
    span.set_attribute(
        "gcp.vertex.agent.llm_request",
        json.dumps({
            "contents": [{
                "role": "user",
                "parts": [{"text": user_text}],
            }],
        }),
    )

  utterances: list[str] = []
  current_text = ""
  current_start: Optional[float] = None

  def _close_utterance() -> None:
    nonlocal current_text, current_start
    if current_text:
      utterances.append(current_text)
      span.add_event(
          "model_utterance",
          attributes={"text": current_text},
          timestamp=int(current_start * 1e9),
      )
    current_text = ""
    current_start = None

  for evt in events:
    if evt.invocation_id != invocation_id or evt.author == _USER_AUTHOR:
      continue
    if evt.get_function_calls():
      _close_utterance()
      for call in evt.get_function_calls():
        span.add_event(
            "tool_call",
            attributes={
                "tool": call.name or "",
                "args": json.dumps(call.args, default=str),
            },
            timestamp=int(evt.timestamp * 1e9),
        )
    elif evt.get_function_responses():
      for response in evt.get_function_responses():
        span.add_event(
            "tool_response",
            attributes={
                "tool": response.name or "",
                "response": json.dumps(response.response, default=str),
            },
            timestamp=int(evt.timestamp * 1e9),
        )
    elif evt.output_transcription and evt.output_transcription.text:
      if current_start is None:
        current_start = evt.timestamp
      if not evt.partial:
        # A non-partial transcription carries the authoritative full text
        # of the current utterance.
        current_text = evt.output_transcription.text
      else:
        current_text += evt.output_transcription.text
  _close_utterance()

  if utterances:
    span.set_attribute(
        "gcp.vertex.agent.llm_response",
        json.dumps({
            "content": {
                "role": "model",
                "parts": [{"text": text} for text in utterances],
            },
        }),
    )


class EvalCaseResponses(BaseModel):
  """Contains multiple responses associated with an EvalCase.

  Multiple responses are a result of repeated requests to generate inferences.
  """

  eval_case: EvalCase
  responses: list[list[Invocation]]


class _LiveSession:
  """Manages the background task and state for a live session."""

  def __init__(
      self,
      runner: Runner,
      session: Session,
      user_id: str,
      session_id: str,
  ):
    self.runner = runner
    self.session = session
    self.user_id = user_id
    self.session_id = session_id
    self.live_request_queue = LiveRequestQueue()
    self.event_queue = asyncio.Queue()
    self.turn_complete_event = asyncio.Event()
    self.live_finished = asyncio.Event()
    self.current_invocation_id = Event.new_id()
    self.consume_task = None
    # OTel context whose current span is the per-turn `live_turn`. Set by
    # the main task before sending a user message and cleared after the
    # turn completes. The consume task attaches it around
    # `handle_function_calls_live` so tool spans are parented under
    # `live_turn` (i.e. live in the same trace as their turn) instead of
    # under whatever ambient context this task happens to have.
    self.current_turn_context: Optional[context_api.Context] = None

  async def __aenter__(self) -> _LiveSession:
    """Starts the background task."""
    self.consume_task = asyncio.create_task(self._consume_events())
    return self

  async def _consume_events(self) -> None:
    """Background task: consume events from run_live."""
    try:
      run_config = RunConfig(
          streaming_mode=StreamingMode.BIDI,
          response_modalities=["AUDIO"],
          output_audio_transcription=types.AudioTranscriptionConfig(),
          input_audio_transcription=types.AudioTranscriptionConfig(),
      )

      invocation_context = self.runner._new_invocation_context_for_live(
          self.session,
          live_request_queue=self.live_request_queue,
          run_config=run_config,
      )
      invocation_context.agent = self.runner._find_agent_to_run(
          self.session, self.runner.agent
      )

      # Run before_agent_callback before any instruction preprocessing.
      # `agent.run_live` (bypassed below) would normally fire this. Without it,
      # an agent that seeds session state in before_agent_callback raises
      # KeyError when `_preprocess_async` renders a `{state_var}` referenced by
      # its instruction template. The callback writes through to
      # `session.state` (State.__setitem__), and we append the resulting event
      # so the state delta is persisted for non-in-memory session services too.
      before_agent_event = (
          await invocation_context.agent._handle_before_agent_callback(
              invocation_context
          )
      )
      if before_agent_event:
        await self.runner.session_service.append_event(
            session=self.session, event=before_agent_event
        )
      if invocation_context.end_invocation:
        return

      callback_context = None
      llm_request = LlmRequest()

      async with Aclosing(
          invocation_context.agent._llm_flow._preprocess_async(
              invocation_context, llm_request
          )
      ) as agen:
        async for _ in agen:
          pass

      callback_context = CallbackContext(invocation_context)
      # By default, live API calls do not include before_model_callback and
      # after_model_callback. These callbacks are needed by the plugins to
      # include the agent instructions and tool declarations in the eval
      # invocations for autorater evaluation.
      await invocation_context.plugin_manager.run_before_model_callback(
          callback_context=callback_context,
          llm_request=llm_request,
      )

      in_function_call_loop = False
      # Bypass `agent.run_live`: it wraps the flow in `record_agent_invocation`
      # which opens a single long-lived `invoke_agent` span covering the
      # entire session. That collapses every turn into one trace and adds an
      # empty/erroring trace to session views (the WebSocket close at
      # session-end gets recorded as a span exception). Call the impl
      # directly; per-turn `live_turn` spans (opened by the main task) take
      # over the role of session-grouped invocation spans for eval purposes.
      # `before_agent_callback` is run explicitly above (it must fire before
      # `_preprocess_async`, and `run_live` would fire it a second time);
      # `after_agent_callback` is still skipped here — fine for evals of
      # agents that don't rely on it.
      async with Aclosing(
          invocation_context.agent._run_live_impl(invocation_context)
      ) as agen:
        # Drive the generator manually so we can attach
        # `current_turn_context` around BOTH the generator's internal work
        # (which also runs `handle_function_calls_live` — see
        # base_llm_flow.py:_receive_from_model) and our own body below.
        # Without this, the first invocation of the tool — the one inside
        # `_run_live_impl` — runs without `live_turn` as parent and lands
        # as an orphan root span.
        while True:
          token = (
              context_api.attach(self.current_turn_context)
              if self.current_turn_context is not None
              else None
          )
          try:
            try:
              event = await agen.__anext__()
            except StopAsyncIteration:
              break
            assert event is not None
            event.invocation_id = self.current_invocation_id
            if callback_context:
              await invocation_context.plugin_manager.run_after_model_callback(
                  callback_context=callback_context,
                  llm_response=event,
              )
            await self.event_queue.put(event)
            if not event.partial:
              await self.runner.session_service.append_event(
                  session=self.session, event=event
              )
            # Track the "function-call → tool-response → final answer"
            # interlude so the first `turn_complete` (which signals "I've
            # issued my tool call") doesn't release the main task — the
            # turn isn't really done until the post-tool reply arrives.
            if event.get_function_calls():
              in_function_call_loop = True

            # `_run_live_impl` (base_llm_flow.py:_receive_from_model)
            # already runs `handle_function_calls_live` and yields the
            # resulting function_response event. Running it again here
            # would execute the tool twice. Just forward the response
            # parts back into the live model.
            if event.content and event.content.parts:
              for part in event.content.parts:
                if part.function_response:
                  tool_content = types.Content(
                      role="tool",
                      parts=[part],
                  )
                  self.live_request_queue.send_content(tool_content)

            if event.turn_complete and event.author != _USER_AUTHOR:
              if not in_function_call_loop:
                self.turn_complete_event.set()
              else:
                in_function_call_loop = False
          finally:
            if token is not None:
              context_api.detach(token)
    finally:
      self.live_finished.set()
      self.turn_complete_event.set()  # Unblock any waiters

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Closes the queue and waits for the background task to finish."""
    self.live_request_queue.close()
    try:
      await asyncio.wait_for(self.consume_task, timeout=30)
    except asyncio.TimeoutError:
      logger.warning("Timed out waiting for run_live to finish.")
      assert self.consume_task is not None
      self.consume_task.cancel()
      try:
        await self.consume_task
      except asyncio.CancelledError:
        pass
    except (ConnectionClosed, errors.APIError) as e:
      # The Gemini Live API uses WebSockets. When the session ends normally, the
      # connection is closed with code 1000. Some client libraries may raise an
      # exception rather than handling it silently. We log this as INFO to
      # avoid false-positive error reports for expected behavior.
      is_normal_closure = isinstance(e, ConnectionClosedOK) or (
          isinstance(e, errors.APIError) and e.code == 1000
      )

      if is_normal_closure:
        logger.info("Ignored WebSocket normal closure exception: %s", e)
      else:
        raise


class EvaluationGenerator:
  """Generates evaluation responses for agents."""

  @staticmethod
  async def generate_responses(
      eval_set: EvalSet,
      agent_module_path: str,
      repeat_num: int = 3,
      agent_name: str = None,
      user_simulator_config: Optional[BaseUserSimulatorConfig] = None,
  ) -> list[EvalCaseResponses]:
    """Returns evaluation responses for the given dataset and agent.

    Args:
      eval_set: The eval set that needs to be scraped for responses.
      agent_module_path: Path to the module that contains the root agent.
      repeat_num: Number of time the eval dataset should be repeated. This is
        usually done to remove uncertainty that a single run may bring.
      agent_name: The name of the agent that should be evaluated. This is
        usually the sub-agent.
      user_simulator_config: Optional configuration for the user simulator.
        Only relevant for eval cases that use a `conversation_scenario` (which
        are driven by `LlmBackedUserSimulator`); ignored for static
        conversations. Pass an `LlmBackedUserSimulatorConfig` to override the
        user-simulation model, max invocations, or custom instructions.
    """
    results = []

    for eval_case in eval_set.eval_cases:
      user_simulator = UserSimulatorProvider(
          user_simulator_config=user_simulator_config
      ).provide(eval_case)

      responses = []
      for _ in range(repeat_num):
        response_invocations = await EvaluationGenerator._process_query(
            agent_module_path,
            user_simulator,
            agent_name,
            eval_case.session_input,
        )
        responses.append(response_invocations)

      results.append(
          EvalCaseResponses(eval_case=eval_case, responses=responses)
      )

    return results

  @staticmethod
  def generate_responses_from_session(session_path, eval_dataset):
    """Returns evaluation responses by combining session data with eval data.

    Args:
      session_path: Path to a json file that contains session data.
      eval_dataset: The eval data set that should be combined with the session
        data.
    """
    results = []

    with open(session_path, "r") as f:
      session_data = Session.model_validate_json(f.read())
      logger.info("Loaded session %s", session_path)

    for data in eval_dataset:
      # load session data from session_path
      results.append(
          EvaluationGenerator._process_query_with_session(
              session_data,
              data,
          )
      )

    return results

  @staticmethod
  def _is_live_api_model(name: str) -> bool:
    """Detects Gemini Live API models by name (e.g. `gemini-live-...`)."""
    return "live" in name

  @staticmethod
  async def _process_query(
      module_name: str,
      user_simulator: UserSimulator,
      agent_name: Optional[str] = None,
      initial_session: Optional[SessionInput] = None,
  ) -> list[Invocation]:
    """Process a query using the agent and evaluation dataset."""
    module_path = f"{module_name}"
    agent_module = importlib.import_module(module_path)
    root_agent = agent_module.agent.root_agent

    reset_func = getattr(agent_module.agent, "reset_data", None)

    agent_to_evaluate = root_agent
    if agent_name:
      agent_to_evaluate = root_agent.find_agent(agent_name)
      assert agent_to_evaluate, f"Sub-Agent `{agent_name}` not found."

    if EvaluationGenerator._is_live_api_model(agent_to_evaluate.model):
      return (
          await EvaluationGenerator._generate_inferences_from_root_agent_live(
              root_agent=agent_to_evaluate,
              user_simulator=user_simulator,
              reset_func=reset_func,
              initial_session=initial_session,
          )
      )

    else:
      return await EvaluationGenerator._generate_inferences_from_root_agent(
          agent_to_evaluate,
          user_simulator=user_simulator,
          reset_func=reset_func,
          initial_session=initial_session,
      )

  @staticmethod
  async def _generate_inferences_for_single_user_invocation(
      runner: Runner,
      user_id: str,
      session_id: str,
      user_content: Content,
  ) -> AsyncGenerator[Event, None]:
    invocation_id = None

    async with Aclosing(
        runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content,
        )
    ) as agen:

      async for event in agen:
        if not invocation_id:
          invocation_id = event.invocation_id
          yield Event(
              content=user_content,
              author=_USER_AUTHOR,
              invocation_id=invocation_id,
          )

        yield event

  @staticmethod
  async def _generate_inferences_for_single_user_invocation_live(
      live_request_queue: LiveRequestQueue,
      event_queue: asyncio.Queue[Event],
      user_message: Content,
      current_invocation_id: str,
      turn_complete_event: asyncio.Event,
      live_timeout_seconds: int,
      agent_name: str = _DEFAULT_AUTHOR,
  ) -> AsyncGenerator[Event, None]:
    """Generates inferences for a single user invocation in live mode."""
    yield Event(
        content=user_message,
        author=_USER_AUTHOR,
        invocation_id=current_invocation_id,
    )

    live_request_queue.send_content(user_message)

    try:
      await asyncio.wait_for(
          turn_complete_event.wait(),
          timeout=live_timeout_seconds,
      )
    except asyncio.TimeoutError:
      logger.warning(
          "Timed out waiting for model turn completion in live mode."
      )
      raise

    # Server-side audio transcription is decoupled from the audio stream:
    # `turn_complete` only marks the end of generation, and the
    # `output_transcription` events can trail it by seconds (the connection
    # flushes whatever fragment it has accumulated when the turn signal
    # arrives, so even a finished=True transcription seen here may be
    # incomplete). Each received event resets the idle window, so a flowing
    # tail keeps the drain alive; the window only has to outlast the gaps.
    while True:
      try:
        event = await asyncio.wait_for(
            event_queue.get(), timeout=_TRANSCRIPTION_TAIL_GRACE_SECONDS
        )
      except asyncio.TimeoutError:
        break
      if event.invocation_id != current_invocation_id:
        logger.debug(
            "Dropped straggler event from invocation %s while draining %s.",
            event.invocation_id,
            current_invocation_id,
        )
        continue
      yield event
      # Emit a synthetic text event for each transcription, preserving
      # the order in which events are received.
      if (
          event.author != _USER_AUTHOR
          and event.output_transcription
          and event.output_transcription.text
          and event.partial
      ):
        yield Event(
            content=Content(
                role="model",
                parts=[types.Part(text=event.output_transcription.text)],
            ),
            author=agent_name,
            invocation_id=current_invocation_id,
            custom_metadata={TRANSCRIPTION_CHUNK_METADATA_KEY: True},
        )

  @staticmethod
  async def _generate_inferences_from_root_agent_live(
      root_agent: Agent,
      user_simulator: UserSimulator,
      reset_func: Optional[Any] = None,
      initial_session: Optional[SessionInput] = None,
      session_id: Optional[str] = None,
      session_service: Optional[BaseSessionService] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      memory_service: Optional[BaseMemoryService] = None,
      live_timeout_seconds: int = DEFAULT_LIVE_TIMEOUT_SECONDS,
  ) -> list[Invocation]:
    """Scrapes the root agent in coordination with the user simulator in live mode."""
    if not session_service:
      session_service = InMemorySessionService()

    if not memory_service:
      memory_service = InMemoryMemoryService()

    app_name = (
        initial_session.app_name if initial_session else "EvaluationGenerator"
    )
    user_id = initial_session.user_id if initial_session else "test_user_id"
    session_id = session_id if session_id else str(uuid.uuid4())

    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        state=initial_session.state if initial_session else {},
        session_id=session_id,
    )

    if not artifact_service:
      artifact_service = InMemoryArtifactService()

    # Reset agent state for each query
    if callable(reset_func):
      reset_func()

    # We ensure that there is some kind of retries on the llm_requests that are
    # generated from the Agent. This is done to make inferencing step of evals
    # more resilient to temporary model failures.
    ensure_retry_options_plugin = EnsureRetryOptionsPlugin(
        name="ensure_retry_options"
    )
    request_intercepter_plugin = _RequestIntercepterPlugin(
        name="request_intercepter_plugin"
    )
    async with Runner(
        app_name=app_name,
        agent=root_agent,
        artifact_service=artifact_service,
        session_service=session_service,
        memory_service=memory_service,
        plugins=[request_intercepter_plugin, ensure_retry_options_plugin],
    ) as runner:
      events = []

      # `_LiveSession` is a runtime connection manager wrapping the `Session`
      # data model (which stores conversation history/state). It manages the
      # active bidirectional WebSocket stream and background consumer tasks.
      live_session = _LiveSession(runner, session, user_id, session_id)
      await live_session.__aenter__()

      try:
        turn_idx = 0
        while True:
          turn_idx += 1
          next_user_message = await user_simulator.get_next_user_message(
              copy.deepcopy(events)
          )
          if next_user_message.status == UserSimulatorStatus.SUCCESS:
            live_session.current_invocation_id = Event.new_id()
            live_session.turn_complete_event.clear()

            logger.info("Waiting for model to complete turn %d...", turn_idx)

            # Open a per-turn root span. By using an empty parent context it
            # becomes the root of its own trace, which session-grouping
            # tracing pipelines (e.g. MLflow Sessions) treat as one chat-turn
            # entry. Tool calls executed during the turn are re-parented
            # under this span by `_LiveSession._consume_events` via
            # `current_turn_context`, so the whole turn lives in one trace.
            live_turn_span = _telemetry.tracer.start_span(
                "live_turn",
                context=context_api.Context(),
                start_time=time.time_ns(),
            )
            live_turn_span.set_attribute("gen_ai.conversation.id", session_id)
            live_turn_span.set_attribute("gen_ai.agent.name", runner.agent.name)
            live_turn_span.set_attribute("gen_ai.operation.name", "chat")
            live_session.current_turn_context = set_span_in_context(
                live_turn_span
            )
            try:
              async for (
                  event
              ) in EvaluationGenerator._generate_inferences_for_single_user_invocation_live(
                  live_request_queue=live_session.live_request_queue,
                  event_queue=live_session.event_queue,
                  user_message=next_user_message.user_message,
                  current_invocation_id=live_session.current_invocation_id,
                  turn_complete_event=live_session.turn_complete_event,
                  live_timeout_seconds=live_timeout_seconds,
                  agent_name=runner.agent.name,
              ):
                events.append(event)

              # The synthetic text events for the eval trajectory are emitted
              # per transcription chunk (in arrival order) by
              # `_generate_inferences_for_single_user_invocation_live`; the
              # span gets per-utterance attributes and timestamped events.
              _record_live_turn_telemetry(
                  live_turn_span,
                  events,
                  live_session.current_invocation_id,
                  next_user_message.user_message,
              )
            finally:
              live_session.current_turn_context = None
              live_turn_span.end()

            if live_session.live_finished.is_set():
              logger.info("Live session finished signal detected.")
              break
          else:  # no message generated
            break
      finally:
        await live_session.__aexit__(None, None, None)

      app_details_by_invocation_id = (
          EvaluationGenerator._get_app_details_by_invocation_id(
              events, request_intercepter_plugin
          )
      )
      return EvaluationGenerator.convert_events_to_eval_invocations(
          events, app_details_by_invocation_id
      )

  @staticmethod
  async def _generate_inferences_from_root_agent(
      root_agent: Agent,
      user_simulator: UserSimulator,
      reset_func: Optional[Any] = None,
      initial_session: Optional[SessionInput] = None,
      session_id: Optional[str] = None,
      session_service: Optional[BaseSessionService] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      memory_service: Optional[BaseMemoryService] = None,
  ) -> list[Invocation]:
    """Scrapes the root agent in coordination with the user simulator."""

    if not session_service:
      session_service = InMemorySessionService()

    if not memory_service:
      memory_service = InMemoryMemoryService()

    app_name = (
        initial_session.app_name if initial_session else "EvaluationGenerator"
    )
    user_id = initial_session.user_id if initial_session else "test_user_id"
    session_id = session_id if session_id else str(uuid.uuid4())

    _ = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        state=initial_session.state if initial_session else {},
        session_id=session_id,
    )

    if not artifact_service:
      artifact_service = InMemoryArtifactService()

    # Reset agent state for each query
    if callable(reset_func):
      reset_func()

    request_intercepter_plugin = _RequestIntercepterPlugin(
        name="request_intercepter_plugin"
    )
    # We ensure that there is some kind of retries on the llm_requests that are
    # generated from the Agent. This is done to make inferencing step of evals
    # more resilient to temporary model failures.
    ensure_retry_options_plugin = EnsureRetryOptionsPlugin(
        name="ensure_retry_options"
    )
    async with Runner(
        app_name=app_name,
        agent=root_agent,
        artifact_service=artifact_service,
        session_service=session_service,
        memory_service=memory_service,
        plugins=[request_intercepter_plugin, ensure_retry_options_plugin],
    ) as runner:
      events = []
      while True:
        next_user_message = await user_simulator.get_next_user_message(
            copy.deepcopy(events)
        )
        if next_user_message.status == UserSimulatorStatus.SUCCESS:
          async for (
              event
          ) in EvaluationGenerator._generate_inferences_for_single_user_invocation(
              runner, user_id, session_id, next_user_message.user_message
          ):
            events.append(event)
        else:  # no message generated
          break

      app_details_by_invocation_id = (
          EvaluationGenerator._get_app_details_by_invocation_id(
              events, request_intercepter_plugin
          )
      )
      return EvaluationGenerator.convert_events_to_eval_invocations(
          events, app_details_by_invocation_id
      )

  @staticmethod
  def convert_events_to_eval_invocations(
      events: list[Event],
      app_details_per_invocation: Optional[dict[str, AppDetails]] = None,
  ) -> list[Invocation]:
    """Converts a list of events to eval invocations."""
    events_by_invocation_id = (
        EvaluationGenerator._collect_events_by_invocation_id(events)
    )

    invocations = []
    for invocation_id, events in events_by_invocation_id.items():
      final_response = None
      final_event = None
      user_content = Content(parts=[])
      invocation_timestamp = 0
      app_details = None
      if (
          app_details_per_invocation
          and invocation_id in app_details_per_invocation
      ):
        app_details = app_details_per_invocation[invocation_id]

      events_to_add = []

      for event in events:
        current_author = (event.author or _DEFAULT_AUTHOR).lower()

        if current_author == _USER_AUTHOR:
          # If the author is the user, then we just identify it and move on
          # to the next event.
          user_content = event.content
          invocation_timestamp = event.timestamp
          continue

        if event.content and event.content.parts:
          if event.is_final_response():
            final_response = event.content
            final_event = event

          for p in event.content.parts:
            if (
                p.function_call
                or p.function_response
                or p.text
                or p.inline_data
            ):
              events_to_add.append(event)
              break

      invocation_events = [
          InvocationEvent(author=e.author, content=e.content)
          for e in events_to_add
          if e is not final_event
      ]
      invocations.append(
          Invocation(
              invocation_id=invocation_id,
              user_content=user_content,
              final_response=final_response,
              intermediate_data=InvocationEvents(
                  invocation_events=invocation_events
              ),
              creation_timestamp=invocation_timestamp,
              app_details=app_details,
          )
      )

    return invocations

  @staticmethod
  def _get_app_details_by_invocation_id(
      events: list[Event], request_intercepter: _RequestIntercepterPlugin
  ) -> dict[str, AppDetails]:
    """Creates an AppDetails object from the list of events."""
    events_by_invocation_id = (
        EvaluationGenerator._collect_events_by_invocation_id(events)
    )
    app_details_by_invocation_id = {}

    for invocation_id, events in events_by_invocation_id.items():
      app_details = AppDetails(agent_details={})
      app_details_by_invocation_id[invocation_id] = app_details

      for event in events:
        if event.author == _USER_AUTHOR:
          continue

        llm_request = request_intercepter.get_model_request(event)

        if not llm_request:
          continue

        if event.author not in app_details.agent_details:
          agent_name = event.author
          app_details.agent_details[agent_name] = AgentDetails(
              name=agent_name,
              instructions=llm_request.config.system_instruction,
              tool_declarations=llm_request.config.tools or [],
          )

    return app_details_by_invocation_id

  @staticmethod
  def _collect_events_by_invocation_id(events: list[Event]) -> dict[str, Event]:
    # Group Events by invocation id. Events that share the same invocation id
    # belong to the same invocation.
    events_by_invocation_id: dict[str, list[Event]] = {}

    for event in events:
      invocation_id = event.invocation_id

      if invocation_id not in events_by_invocation_id:
        events_by_invocation_id[invocation_id] = []

      events_by_invocation_id[invocation_id].append(event)

    return events_by_invocation_id

  @staticmethod
  def _process_query_with_session(session_data, data):
    """Process the queries using the existing session data without invoking the runner."""
    responses = data.copy()

    # Iterate through the provided queries and align them with the session
    # events
    for index, eval_entry in enumerate(responses):
      query = eval_entry["query"]
      actual_tool_uses = []
      response = None

      # Search for the corresponding session events
      for event in session_data.events:
        # Match the query to a user event
        if (
            event.author == "user"
            and event.content
            and event.content.parts
            and event.content.parts[0].text == query
        ):
          # Look for subsequent tool usage or model responses
          for subsequent_event in session_data.events:
            if subsequent_event.invocation_id == event.invocation_id:
              # Extract tool usage
              if subsequent_event.content.parts[0].function_call:
                call = subsequent_event.content.parts[0].function_call
                actual_tool_uses.append(
                    {"tool_name": call.name, "tool_input": call.args}
                )
              # Extract final response
              elif subsequent_event.author != "user":
                response = subsequent_event.content.parts[0].text

      # Update the results for the current query
      responses[index]["actual_tool_use"] = actual_tool_uses
      responses[index]["response"] = response
    return responses
