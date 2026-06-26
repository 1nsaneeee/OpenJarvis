"""Simplified ConversationManager for MVP walking skeleton.

States: IDLE → LISTENING → THINKING → (EXECUTING)* → RESPONDING → IDLE
"""
from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import structlog

from openjarvis.bus.client import BusClient
from openjarvis.bus.schemas import (
    AsrFinal,
    ConvState,
    ConvStateEvent,
    WakeEvent,
)
from openjarvis.llm.base import BaseProvider, Message, ProviderError, ToolCall, ToolSpec
from openjarvis.tools.executor import ToolExecutor

_logger = structlog.get_logger(__name__)


class State(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    EXECUTING = "executing"
    RESPONDING = "responding"


class ConversationManager:
    def __init__(
        self,
        bus: BusClient,
        provider: BaseProvider,
        executor: ToolExecutor,
        *,
        model: str,
        max_history: int = 20,
        max_tool_rounds: int = 5,
        system_prompt_file: str = "config/prompts/system.md",
        tools: list[ToolSpec] | None = None,
    ) -> None:
        self._bus = bus
        self._provider = provider
        self._executor = executor
        self._model = model
        self._max_history = max_history
        self._max_tool_rounds = max_tool_rounds
        self._tools = tools or []
        self._state = State.IDLE
        self._trace_id = ""
        self._history: list[Message] = []
        self._system_prompt = self._load_system_prompt(system_prompt_file)

    @staticmethod
    def _load_system_prompt(path: str) -> str:
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return "You are Jarvis, an AI assistant."

    @property
    def state(self) -> State:
        return self._state

    async def start(self) -> None:
        await self._bus.subscribe("jarvis:wake:detected", WakeEvent, self._on_wake)
        await self._bus.subscribe("jarvis:asr:final", AsrFinal, self._on_asr_final)

    async def _set_state(self, new_state: State) -> None:
        old = self._state
        self._state = new_state
        ev = ConvStateEvent(
            source="conversation",
            trace_id=self._trace_id,
            state=ConvState(new_state.value),
            previous=ConvState(old.value),
        )
        await self._bus.publish("jarvis:conv:state", ev)
        await self._bus.set_state("jarvis:conv:state", ConvState(new_state.value))

    def _trim_history(self) -> None:
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    async def _on_wake(self, event: WakeEvent) -> None:
        if self._state != State.IDLE:
            return
        self._trace_id = event.id
        await self._set_state(State.LISTENING)
        print("\n[Jarvis] Listening...")

    async def _on_asr_final(self, event: AsrFinal) -> None:
        if self._state != State.LISTENING:
            return
        print(f"[You] {event.text}")
        self._history.append(Message(role="user", content=event.text))
        self._trim_history()
        await self._think()

    async def _think(self, _tool_round: int = 0) -> None:
        if _tool_round > self._max_tool_rounds:
            await self._set_state(State.RESPONDING)
            print("\n[Jarvis] I had trouble completing that with tools. Please try again.\n")
            await self._set_state(State.IDLE)
            return

        await self._set_state(State.THINKING)
        messages = [Message(role="system", content=self._system_prompt)] + self._history

        full_text = ""
        pending_calls: list[ToolCall] = []

        try:
            async for delta in self._provider.chat(
                messages,
                tools=self._tools or None,
                model=self._model,
            ):
                if delta.text:
                    full_text += delta.text
                if delta.tool_call:
                    pending_calls.append(delta.tool_call)
        except ProviderError as exc:
            _logger.warning("llm_provider_error", error=str(exc))
            print(f"\n[Jarvis] LLM error: {exc}. Returning to idle.\n")
            await self._set_state(State.IDLE)
            return
        except Exception as exc:  # noqa: BLE001
            _logger.exception("conversation_unexpected_error", error=str(exc))
            print("\n[Jarvis] Something went wrong. Returning to idle.\n")
            await self._set_state(State.IDLE)
            return

        if pending_calls:
            self._history.append(
                Message(role="assistant", content=None, tool_calls=pending_calls)
            )
            self._trim_history()
            await self._execute_tools(pending_calls, _tool_round + 1)
        else:
            self._history.append(Message(role="assistant", content=full_text))
            self._trim_history()
            await self._respond(full_text)

    async def _execute_tools(self, calls: list[ToolCall], tool_round: int) -> None:
        await self._set_state(State.EXECUTING)
        for tc in calls:
            _, result_json = await self._executor.execute(tc.name, tc.arguments)
            self._history.append(
                Message(
                    role="tool",
                    content=result_json,
                    tool_call_id=tc.id,
                    name=tc.name,
                )
            )
            self._trim_history()
        await self._think(_tool_round=tool_round)

    async def _respond(self, text: str) -> None:
        await self._set_state(State.RESPONDING)
        print(f"\n[Jarvis] {text}\n")
        await self._set_state(State.IDLE)


__all__ = ["ConversationManager", "State"]
