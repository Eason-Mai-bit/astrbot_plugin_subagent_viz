"""Sub-agent visualization dashboard for AstrBot.

Captures sub-agent (handoff) runs and exposes them to a Plugin Pages dashboard:
a tree, a Gantt timeline and a per-agent conversation view.

Capture strategy
----------------
AstrBot runs sub-agents through ``Context.tool_loop_agent`` *without* passing
``agent_hooks``, and it throws away every intermediate ``AgentResponse`` inside
that method (``async for _ in agent_runner.step_until_done(...)``). So neither
the response stream nor the plugin event bus can see a sub-agent.

What *is* still reachable from a plugin is the keyword arguments: ``
tool_loop_agent`` reads ``agent_hooks`` out of ``**kwargs``. This plugin
therefore installs two narrow patches:

1. ``FunctionToolExecutor._execute_handoff`` - marks the upcoming run as a
   sub-agent and derives its identity (agent name, goal, parent, depth) from
   the live handoff context. This is also the only reliable way to tell a
   sub-agent run apart from the main agent run, which goes through the very
   same ``tool_loop_agent`` entry point.
2. ``Context.tool_loop_agent`` - for marked runs only, injects a
   ``CaptureHooks`` instance and times the run.

``CaptureHooks`` then records the user prompt, reasoning, assistant text, every
tool call with its arguments, every tool result and the final usage numbers.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from astrbot.api import logger
from astrbot.api.star import Context, Star
from astrbot.api.web import json_response, request

try:  # AstrBot internals; absent only if the core layout ever changes.
    from astrbot.core.agent.hooks import BaseAgentRunHooks as _BaseAgentRunHooks
except Exception:  # pragma: no cover - duck typing keeps the hooks working anyway
    _BaseAgentRunHooks = object  # type: ignore[assignment,misc]

# Must equal metadata.yaml's `name` (and the plugin directory name).
#
# The dashboard never builds the API path itself: the host webui constructs
# `/api/v1/plugins/extensions/{plugin_name}/{endpoint}` from the registered
# plugin name, and the backend matches that path against our registered routes
# with re.fullmatch (dashboard/api/plugins.py). A hyphen here while metadata
# uses an underscore makes every route unmatchable ("未找到该路由").
# The same string is also the config-file stem (data/config/{PLUGIN_NAME}_config.json),
# so it must stay in lockstep with metadata.yaml too.
PLUGIN_NAME = "subagent_viz"

# Keys declared in _conf_schema.json; anything else in the config file is ignored.
_CONFIG_KEYS = (
    "max_history",
    "max_turns_per_agent",
    "max_turn_chars",
    "stale_run_seconds",
    "auto_save",
    "demo_on_start",
)

# Limits. Overwritten from the plugin config during initialize(); kept as module
# globals because _clip() is a module-level helper.
MAX_AGENTS = 300
MAX_TURNS_PER_AGENT = 400
MAX_TURN_CHARS = 4000
STALE_RUN_SECONDS = 900

LONG_POLL_SECONDS = 25.0
STALE_CHECK_INTERVAL = 30.0

# Set by the patched _execute_handoff, read by the patched tool_loop_agent.
# Holds the identity of the sub-agent whose run is about to start.
_subagent_ctx: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "subagent_viz_ctx", default=None
)

# Set by the patched _do_handoff_background. A background handoff owns its own
# asyncio task, so it is the only case where cancelling is safe; a foreground
# handoff shares the main agent's task and must be stopped cooperatively.
_handoff_background: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "subagent_viz_background", default=False
)

# Set by the patched tool_loop_agent for the duration of a sub-agent run and read
# by the patched ToolLoopAgentRunner.step_until_done. That method is where the
# per-token AgentResponse stream exists - Context.tool_loop_agent throws it away
# with `async for _ in ...: pass` - so it has to be intercepted from the inside.
_active_capture: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "subagent_viz_active", default=None
)

# Streaming deltas arrive per token. Bumping the change counter that often would
# hammer every open dashboard, so text is buffered and flushed on a timer.
STREAM_FLUSH_SECONDS = 0.12


class AgentStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


TERMINAL_STATUSES = (
    AgentStatus.COMPLETED,
    AgentStatus.FAILED,
    AgentStatus.INTERRUPTED,
)


STOPPED_REASON = "已手动停止"


def _stopped_response() -> Any:
    """A benign final response so a stopped sub-agent does not look like a
    crashed tool call to the main agent."""
    try:
        from astrbot.core.provider.entities import LLMResponse

        return LLMResponse(role="assistant", completion_text="[sub-agent stopped by user]")
    except Exception:
        return None


def _clip(text: str) -> tuple[str, bool]:
    """Clamp a turn body, reporting whether anything was dropped."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if len(text) <= MAX_TURN_CHARS:
        return text, False
    return text[:MAX_TURN_CHARS], True


def _tool_result_text(tool_result: Any) -> tuple[str, bool]:
    """Flatten an mcp CallToolResult into displayable text.

    Kept for the demo/back-compat path: live runs get their results from the
    runner stream instead, which already carries plain text.
    """
    if tool_result is None:
        return "", False
    parts: list[str] = []
    for item in getattr(tool_result, "content", None) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
        elif getattr(item, "type", "") == "image":
            parts.append("[image]")
    return _clip("\n".join(parts))


@dataclass
class Turn:
    """One entry in a sub-agent's conversation."""

    seq: int
    role: str  # system | user | assistant | reasoning | tool_call | tool_result | error
    text: str = ""
    ts: float = 0.0
    tool: Optional[str] = None
    args: Optional[dict] = None
    is_error: bool = False
    truncated: bool = False
    streaming: bool = False
    """True while this turn is still being appended to."""

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "role": self.role,
            "text": self.text,
            "ts": self.ts,
            "tool": self.tool,
            "args": self.args,
            "is_error": self.is_error,
            "truncated": self.truncated,
            "streaming": self.streaming,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Turn":
        return cls(
            seq=int(raw.get("seq", 0)),
            role=str(raw.get("role", "assistant")),
            text=str(raw.get("text", "")),
            ts=float(raw.get("ts", 0.0)),
            tool=raw.get("tool"),
            args=raw.get("args"),
            is_error=bool(raw.get("is_error", False)),
            truncated=bool(raw.get("truncated", False)),
            streaming=bool(raw.get("streaming", False)),
        )


@dataclass
class SubagentProgress:
    id: str
    name: str = ""
    parent_id: Optional[str] = None
    goal: str = ""
    model: Optional[str] = None
    status: AgentStatus = AgentStatus.RUNNING
    depth: int = 0
    started_at: float = 0.0
    ended_at: Optional[float] = None
    updated_at: float = 0.0
    tool_count: int = 0
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None
    tools: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    summary: Optional[str] = None
    error: Optional[str] = None
    current_tool: Optional[str] = None
    turns: list[Turn] = field(default_factory=list)
    is_demo: bool = False

    def to_dict(self, *, with_turns: bool = False) -> dict:
        data = {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "goal": self.goal,
            "model": self.model,
            "status": self.status.value,
            "depth": self.depth,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "updated_at": self.updated_at,
            "duration_seconds": self.duration,
            "tool_count": self.tool_count,
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "tools": list(self.tools),
            "files_read": list(self.files_read),
            "files_written": list(self.files_written),
            "summary": self.summary,
            "error": self.error,
            "current_tool": self.current_tool,
            "turn_count": len(self.turns),
            "is_demo": self.is_demo,
        }
        if with_turns:
            data["turns"] = [t.to_dict() for t in self.turns]
        return data

    @property
    def duration(self) -> Optional[float]:
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, end - self.started_at) if self.started_at else None

    def to_storage(self) -> dict:
        data = self.to_dict(with_turns=True)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_storage(cls, raw: dict) -> "SubagentProgress":
        def as_int(key: str, fallback: int = 0) -> int:
            try:
                value = raw.get(key, fallback)
                return int(value) if value is not None else fallback
            except (TypeError, ValueError):
                return fallback

        def as_float(key: str, fallback: float = 0.0) -> float:
            try:
                value = raw.get(key, fallback)
                return float(value) if value is not None else fallback
            except (TypeError, ValueError):
                return fallback

        def as_opt_float(key: str) -> Optional[float]:
            value = raw.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        try:
            status = AgentStatus(raw.get("status", "completed"))
        except ValueError:
            status = AgentStatus.COMPLETED

        agent = cls(
            id=str(raw.get("id", "")),
            name=str(raw.get("name", "") or ""),
            parent_id=raw.get("parent_id"),
            goal=str(raw.get("goal", "") or ""),
            model=raw.get("model"),
            status=status,
            depth=as_int("depth"),
            started_at=as_float("started_at"),
            ended_at=as_opt_float("ended_at"),
            updated_at=as_float("updated_at"),
            tool_count=as_int("tool_count"),
            api_calls=as_int("api_calls"),
            input_tokens=as_int("input_tokens"),
            output_tokens=as_int("output_tokens"),
            cost_usd=as_opt_float("cost_usd"),
            tools=[str(t) for t in (raw.get("tools") or [])],
            files_read=[str(t) for t in (raw.get("files_read") or [])],
            files_written=[str(t) for t in (raw.get("files_written") or [])],
            summary=raw.get("summary"),
            error=raw.get("error"),
            is_demo=bool(raw.get("is_demo", False)),
        )
        agent.turns = [Turn.from_dict(t) for t in (raw.get("turns") or [])]
        return agent


class _SubagentStop(BaseException):
    """Raised from a capture hook to abort one sub-agent run.

    Deliberately a ``BaseException``: AstrBot wraps every agent hook call in
    ``except Exception`` (``tool_loop_agent_runner.py``), so an ordinary
    exception raised here would be logged and swallowed and the run would
    carry on regardless.
    """


class CaptureHooks(_BaseAgentRunHooks):
    """Bookkeeping for one sub-agent run.

    The conversation itself is recorded from the runner's response stream (see
    ``_install_patches``), which carries richer data - resolved tool arguments,
    partial text, token usage. These hooks are kept for two things the stream
    cannot provide: the cooperative stop checkpoint, and the final response.
    """

    def __init__(self, plugin: "SubagentVizPlugin", agent_id: str, inner: Any = None):
        self._plugin = plugin
        self._agent_id = agent_id
        self._inner = inner

    async def on_agent_begin(self, run_context: Any) -> None:
        self._plugin._checkpoint(self._agent_id)
        if self._inner is not None:
            await self._inner.on_agent_begin(run_context)

    async def on_tool_start(
        self, run_context: Any, tool: Any, tool_args: dict | None
    ) -> None:
        # only a stop checkpoint - tool calls are recorded from the stream
        self._plugin._checkpoint(self._agent_id)
        if self._inner is not None:
            await self._inner.on_tool_start(run_context, tool, tool_args)

    async def on_tool_end(
        self,
        run_context: Any,
        tool: Any,
        tool_args: dict | None,
        tool_result: Any,
    ) -> None:
        self._plugin._checkpoint(self._agent_id)
        if self._inner is not None:
            await self._inner.on_tool_end(run_context, tool, tool_args, tool_result)

    async def on_agent_done(self, run_context: Any, llm_response: Any) -> None:
        self._plugin._record_final(self._agent_id, llm_response)
        if self._inner is not None:
            await self._inner.on_agent_done(run_context, llm_response)


class SubagentVizPlugin(Star):
    def __init__(self, context: Context, config: Any = None):
        # The host injects the schema-validated AstrBotConfig here
        # (star_manager.py instantiates with config=plugin_config). Accepting it
        # also avoids the TypeError fallback path in the loader.
        super().__init__(context)
        self.config = config
        self._agents: dict[str, SubagentProgress] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._main_tasks: set[asyncio.Task] = set()
        self._stop_requested: set[str] = set()
        self._version = 0
        self._waiters: set[asyncio.Event] = set()
        self._patches: list[tuple[Any, str, Any]] = []
        self._save_pending = False
        self._load_error: Optional[str] = None
        self._auto_save = True
        self._demo_on_start = False
        self._stale_task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._stream_dirty = False
        self._store_path = self._resolve_store_path()

    # ── lifecycle ───────────────────────────────────────────────────────

    async def initialize(self) -> None:
        self._apply_config()
        self._load_from_disk()
        self._register_apis()
        self._install_patches()
        if self._demo_on_start:
            self._load_demo_data()
        self._stale_task = asyncio.create_task(self._stale_watcher())
        logger.info(
            "[subagent-viz] ready: %d restored agent(s), store=%s",
            len(self._agents),
            self._store_path,
        )

    async def terminate(self) -> None:
        for name in ("_stale_task", "_flush_task"):
            task = getattr(self, name, None)
            if task is not None:
                task.cancel()
        self._remove_patches()
        if self._auto_save:
            self._save_to_disk()
        self._agents.clear()
        self._tasks.clear()

    def _apply_config(self) -> None:
        """Honour _conf_schema.json, falling back to the module defaults."""
        global MAX_AGENTS, MAX_TURNS_PER_AGENT, MAX_TURN_CHARS, STALE_RUN_SECONDS
        raw = {}
        # Prefer the AstrBotConfig the host injects from
        # data/config/{dir_name}_config.json. The legacy astrbot.core.star.config
        # .load_config() helper is NOT usable here: it looks for "{name}.json"
        # (no _config suffix) and expects a nested {key: {"value": ...}} shape,
        # so it always returned False and every setting silently fell back to
        # its default.
        try:
            injected = getattr(self, "config", None)
            if injected is not None:
                raw = {k: injected.get(k) for k in _CONFIG_KEYS if k in injected}
        except Exception as exc:
            logger.debug("[subagent-viz] injected config unusable: %s", exc)

        if not raw:
            # Last resort: read the host-managed config file directly, in the
            # flat shape it is actually stored in.
            try:
                from astrbot.core.utils.astrbot_path import get_astrbot_data_path

                cfg = Path(get_astrbot_data_path()) / "config" / f"{PLUGIN_NAME}_config.json"
                if cfg.is_file():
                    raw = {
                        k: v
                        for k, v in (json.loads(cfg.read_text(encoding="utf-8-sig")) or {}).items()
                        if k in _CONFIG_KEYS
                    }
            except Exception as exc:
                logger.debug("[subagent-viz] config file unavailable: %s", exc)

        def as_int(key: str, fallback: int, minimum: int = 0) -> int:
            try:
                return max(minimum, int(raw.get(key, fallback)))
            except (TypeError, ValueError):
                return fallback

        MAX_AGENTS = as_int("max_history", 300, 10)
        MAX_TURNS_PER_AGENT = as_int("max_turns_per_agent", 400, 10)
        MAX_TURN_CHARS = as_int("max_turn_chars", 4000, 200)
        STALE_RUN_SECONDS = as_int("stale_run_seconds", 900, 0)
        self._auto_save = bool(raw.get("auto_save", True))
        self._demo_on_start = bool(raw.get("demo_on_start", False))
        self._dirty = False

    async def _stale_watcher(self) -> None:
        """Flag runs that stopped reporting without ever reaching a terminal state.

        A crashed or wedged task can leave a row stuck on "running" forever; this
        is the only safety net for that, since the capture path's own handlers
        cannot run once the task is gone.
        """
        try:
            while True:
                await asyncio.sleep(STALE_CHECK_INTERVAL)
                if STALE_RUN_SECONDS <= 0:
                    continue
                cutoff = time.time() - STALE_RUN_SECONDS
                changed = False
                for agent in list(self._agents.values()):
                    if agent.status in TERMINAL_STATUSES:
                        continue
                    if agent.updated_at and agent.updated_at < cutoff:
                        self._finish_agent(
                            agent.id,
                            AgentStatus.INTERRUPTED,
                            error=f"超过 {STALE_RUN_SECONDS}s 没有更新，判定为中断",
                        )
                        changed = True
                if changed and self._auto_save:
                    self._schedule_save()
        except asyncio.CancelledError:
            pass

    def _resolve_store_path(self) -> Path:
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path

            base = Path(get_astrbot_data_path())
        except Exception:
            base = Path(__file__).resolve().parent
        # Deliberately NOT PLUGIN_NAME: this on-disk path predates the rename
        # to subagent_viz and already holds captured run history, so it stays
        # frozen to avoid orphaning existing data.
        return base / "plugin_data" / "subagent-viz" / "runs.json"

    # ── patching ────────────────────────────────────────────────────────

    def _install_patches(self) -> None:
        if self._patches:
            return
        try:
            from astrbot.core.agent.runners.tool_loop_agent_runner import (
                ToolLoopAgentRunner,
            )
            from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
            from astrbot.core.star.context import Context as AstrContext
        except Exception as exc:  # pragma: no cover - depends on AstrBot internals
            logger.error("[subagent-viz] cannot import AstrBot internals: %s", exc)
            return

        plugin = self

        # Save the raw class attributes (a classmethod object and a plain
        # function) so teardown restores them exactly, instead of leaving a
        # bound method behind on the class.
        original_handoff_attr = FunctionToolExecutor.__dict__["_execute_handoff"]
        original_loop_attr = AstrContext.__dict__["tool_loop_agent"]
        original_do_bg_attr = FunctionToolExecutor.__dict__["_do_handoff_background"]
        original_steps_attr = ToolLoopAgentRunner.__dict__["step_until_done"]
        # ...and the resolved versions, for calling through the wrapper.
        original_handoff = FunctionToolExecutor._execute_handoff
        original_loop = AstrContext.tool_loop_agent
        original_do_bg = FunctionToolExecutor._do_handoff_background
        original_steps = ToolLoopAgentRunner.step_until_done

        async def patched_steps(runner_self, *args, **kwargs):
            # The per-token AgentResponse stream is generated here and then
            # discarded by Context.tool_loop_agent, so this is the one place a
            # plugin can still observe a live sub-agent. Only runs marked by
            # _active_capture (i.e. our own patched tool_loop_agent) are teed;
            # every other agent in the process passes straight through.
            agent_id = _active_capture.get()
            if agent_id is None:
                async for resp in original_steps(runner_self, *args, **kwargs):
                    yield resp
                return
            async for resp in original_steps(runner_self, *args, **kwargs):
                try:
                    plugin._consume_response(agent_id, resp)
                except Exception as exc:  # never break the agent run
                    logger.debug("[subagent-viz] stream capture error: %s", exc)
                yield resp
            agent = plugin._agents.get(agent_id)
            if agent is not None:
                plugin._close_stream_turn(agent, "assistant")
                plugin._close_stream_turn(agent, "reasoning")
                plugin._touch()

        @classmethod
        async def patched_do_handoff_bg(cls, tool, run_context, **kw):
            # a plain coroutine upstream, not an async generator
            token = _handoff_background.set(True)
            try:
                return await original_do_bg(tool, run_context, **kw)
            finally:
                _handoff_background.reset(token)

        @classmethod
        async def patched_handoff(cls, tool, run_context, **tool_args):
            agent = getattr(tool, "agent", None)
            if agent is None:
                async for item in original_handoff(tool, run_context, **tool_args):
                    yield item
                return

            parent = _subagent_ctx.get()
            meta = {
                "id": f"{getattr(agent, 'name', 'subagent')}-{uuid.uuid4().hex[:8]}",
                "name": getattr(agent, "name", "") or "subagent",
                "goal": str(tool_args.get("input") or ""),
                "parent_id": parent["id"] if parent else None,
                "depth": (parent["depth"] + 1) if parent else 0,
                "background": _handoff_background.get(),
            }
            token = _subagent_ctx.set(meta)
            try:
                async for item in original_handoff(tool, run_context, **tool_args):
                    yield item
            finally:
                _subagent_ctx.reset(token)

        async def patched_loop(self_ctx, **kwargs):
            meta = _subagent_ctx.get()
            if meta is None:
                # Main-agent run (or a plugin-initiated one): stay out of the
                # way, but remember the driving task so a later "stop" never
                # cancels the whole conversation turn by mistake.
                task = asyncio.current_task()
                if task is not None:
                    self._main_tasks.add(task)
                try:
                    return await original_loop(self_ctx, **kwargs)
                finally:
                    if task is not None:
                        self._main_tasks.discard(task)

            agent_id = meta["id"]
            inner = kwargs.get("agent_hooks")
            kwargs["agent_hooks"] = CaptureHooks(plugin, agent_id, inner)

            self._begin_agent(meta, kwargs.get("chat_provider_id"), kwargs.get("system_prompt"))
            # Only a background handoff owns its task, so only that one may be
            # cancelled outright. A foreground handoff shares the main agent's
            # task - cancelling it would abort the whole conversation turn.
            task = asyncio.current_task()
            if task is not None and meta.get("background"):
                self._tasks[agent_id] = task
            capture_token = _active_capture.set(agent_id)
            try:
                result = await original_loop(self_ctx, **kwargs)
            except _SubagentStop:
                # Cooperative stop: only this sub-agent's loop unwinds and the
                # main agent receives a normal (empty) answer instead of a
                # crashed tool call.
                self._finish_agent(agent_id, AgentStatus.INTERRUPTED, error=STOPPED_REASON)
                return _stopped_response()
            except asyncio.CancelledError:
                self._finish_agent(agent_id, AgentStatus.INTERRUPTED, error=STOPPED_REASON)
                raise
            except Exception as exc:
                self._finish_agent(agent_id, AgentStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
                logger.warning("[subagent-viz] sub-agent %s failed: %s", meta["name"], exc)
                raise
            else:
                self._finish_agent(agent_id, AgentStatus.COMPLETED)
                return result
            finally:
                _active_capture.reset(capture_token)
                self._tasks.pop(agent_id, None)
                self._stop_requested.discard(agent_id)
                self._close_streams(agent_id)
                self._touch()

        FunctionToolExecutor._execute_handoff = patched_handoff
        FunctionToolExecutor._do_handoff_background = patched_do_handoff_bg
        AstrContext.tool_loop_agent = patched_loop
        ToolLoopAgentRunner.step_until_done = patched_steps
        self._patches = [
            (FunctionToolExecutor, "_execute_handoff", original_handoff_attr),
            (FunctionToolExecutor, "_do_handoff_background", original_do_bg_attr),
            (AstrContext, "tool_loop_agent", original_loop_attr),
            (ToolLoopAgentRunner, "step_until_done", original_steps_attr),
        ]
        self._start_flush_loop()
        logger.info("[subagent-viz] capture patches installed")

    def _remove_patches(self) -> None:
        for owner, attr, original in self._patches:
            try:
                setattr(owner, attr, original)
            except Exception as exc:
                logger.error("[subagent-viz] could not restore %s: %s", attr, exc)
        self._patches.clear()

    # ── capture ─────────────────────────────────────────────────────────

    def _begin_agent(
        self, meta: dict, provider_id: Optional[str], system_prompt: Optional[str]
    ) -> None:
        now = time.time()
        agent = SubagentProgress(
            id=meta["id"],
            name=meta["name"],
            parent_id=meta["parent_id"],
            goal=meta["goal"],
            model=provider_id,
            status=AgentStatus.RUNNING,
            depth=meta["depth"],
            started_at=now,
            updated_at=now,
        )
        self._agents[meta["id"]] = agent
        if system_prompt:
            text, truncated = _clip(system_prompt)
            self._append_turn(agent, "system", text, truncated=truncated)
        self._append_turn(agent, "user", meta["goal"])
        self._evict()
        self._touch()

    def _finish_agent(
        self,
        agent_id: str,
        status: AgentStatus,
        error: Optional[str] = None,
    ) -> None:
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        agent.status = status
        agent.ended_at = time.time()
        agent.updated_at = agent.ended_at
        agent.current_tool = None
        if error:
            agent.error = error
            text, truncated = _clip(error)
            self._append_turn(agent, "error", text, truncated=truncated)
        if not agent.summary:
            for turn in reversed(agent.turns):
                if turn.role == "assistant" and turn.text.strip():
                    agent.summary = turn.text.strip()[:400]
                    break
        self._touch()
        self._schedule_save()

    def _append_turn(
        self,
        agent: SubagentProgress,
        role: str,
        text: str,
        *,
        tool: Optional[str] = None,
        args: Optional[dict] = None,
        is_error: bool = False,
        truncated: bool = False,
    ) -> None:
        if not text and role not in ("tool_call", "tool_result"):
            return
        agent.turns.append(
            Turn(
                seq=len(agent.turns),
                role=role,
                text=text,
                ts=time.time(),
                tool=tool,
                args=args,
                is_error=is_error,
                truncated=truncated,
            )
        )
        if len(agent.turns) > MAX_TURNS_PER_AGENT:
            del agent.turns[: len(agent.turns) - MAX_TURNS_PER_AGENT]
            for index, turn in enumerate(agent.turns):
                turn.seq = index
        agent.updated_at = time.time()

    def _close_streams(self, agent_id: str) -> None:
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        self._close_stream_turn(agent, "assistant")
        self._close_stream_turn(agent, "reasoning")
        agent.current_tool = None
        self._stream_dirty = False

    def _record_final(self, agent_id: str, llm_response: Any) -> None:
        """Final response bookkeeping.

        The text and reasoning already arrived over the stream, so only the
        summary and the usage totals are filled in here - appending the text
        again would duplicate it.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        final_text = ""
        try:
            final_text = llm_response.completion_text or ""
        except Exception:
            final_text = ""
        if final_text.strip():
            agent.summary = final_text.strip()[:400]
            already = any(
                t.role == "assistant" and t.text.strip() for t in agent.turns
            )
            if not already:
                text, truncated = _clip(final_text)
                self._append_turn(agent, "assistant", text, truncated=truncated)
        usage = getattr(llm_response, "usage", None)
        if usage is not None and not (agent.input_tokens or agent.output_tokens):
            try:
                agent.input_tokens += int(getattr(usage, "input", 0) or 0)
                agent.output_tokens += int(getattr(usage, "output", 0) or 0)
            except Exception:
                pass
        self._touch()

    def _checkpoint(self, agent_id: str) -> None:
        """Hook-side guard: raise to abort just this sub-agent when stopped.

        Called at every hook boundary, which is the only place inside the run
        where the plugin regains control.
        """
        if agent_id in self._stop_requested:
            raise _SubagentStop()

    # ── streaming capture ───────────────────────────────────────────────

    @staticmethod
    def _resp_chain(response: Any) -> Any:
        """Get the MessageChain out of an AgentResponse.

        AgentResponseData is a TypedDict, so ``response.data`` is a plain dict
        in practice, not an object - both shapes are handled.
        """
        data = getattr(response, "data", None)
        if isinstance(data, dict):
            return data.get("chain")
        return getattr(data, "chain", None)

    @classmethod
    def _chain_text(cls, response: Any) -> tuple[str, str]:
        """Pull (kind, text) out of an AgentResponse's MessageChain.

        ``kind`` is "reasoning" for thinking deltas and "text" otherwise.
        """
        chain = cls._resp_chain(response)
        if chain is None:
            return "", ""
        kind = "reasoning" if getattr(chain, "type", "") == "reasoning" else "text"
        try:
            text = chain.get_plain_text() or ""
        except Exception:
            text = ""
        return kind, text

    @classmethod
    def _chain_components(cls, response: Any) -> list:
        chain = cls._resp_chain(response)
        components = getattr(chain, "chain", None)
        if components is None and isinstance(chain, (list, tuple)):
            components = chain
        return list(components or [])

    @classmethod
    def _chain_json(cls, response: Any) -> Optional[dict]:
        """Pull the Json payload out of a tool_call / tool_call_result chain."""
        for comp in cls._chain_components(response):
            data = getattr(comp, "data", None)
            if isinstance(data, dict):
                return data
            if isinstance(comp, dict) and isinstance(comp.get("data"), dict):
                return comp["data"]
        return None

    def _stream_text(self, agent_id: str, kind: str, text: str) -> None:
        """Append a token delta to the open turn of the matching kind."""
        if not text:
            return
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        role = "reasoning" if kind == "reasoning" else "assistant"
        turns = agent.turns
        if turns and turns[-1].role == role and turns[-1].streaming:
            turns[-1].text += text
            turns[-1].streaming = True
        else:
            turns.append(
                Turn(seq=len(turns), role=role, text=text, ts=time.time(), streaming=True)
            )
            self._trim_turns(agent)
        self._mark_stream_dirty(agent)

    def _close_stream_turn(self, agent: SubagentProgress, role: str) -> None:
        """Close the newest still-open turn of ``role``.

        Not necessarily ``turns[-1]``: a reasoning turn is usually followed by
        the assistant turn that continues after it, so scanning backwards is
        what actually leaves no turn stuck showing a live cursor.
        """
        for turn in reversed(agent.turns):
            if turn.role == role and turn.streaming:
                turn.streaming = False
                if len(turn.text) > MAX_TURN_CHARS:
                    turn.text = turn.text[:MAX_TURN_CHARS]
                    turn.truncated = True
                return

    def _stream_tool_call(self, agent_id: str, payload: dict) -> None:
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        name = str(payload.get("name") or "tool")
        args = payload.get("args")
        self._close_stream_turn(agent, "assistant")
        self._close_stream_turn(agent, "reasoning")
        agent.turns.append(
            Turn(
                seq=len(agent.turns),
                role="tool_call",
                text="",
                ts=float(payload.get("ts") or time.time()),
                tool=name,
                args=self._safe_json(args) if isinstance(args, dict) else None,
            )
        )
        self._trim_turns(agent)
        agent.tools.append(name)
        agent.tool_count += 1
        agent.api_calls += 1
        agent.current_tool = name
        if isinstance(args, dict):
            self._note_file_access(agent, name, args)
        self._touch()

    def _stream_tool_result(self, agent_id: str, payload: dict) -> None:
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        raw = payload.get("result")
        text, truncated = _clip(raw if isinstance(raw, str) else str(raw or ""))
        name = self._tool_name_for_call(agent) or ""
        agent.turns.append(
            Turn(
                seq=len(agent.turns),
                role="tool_result",
                text=text,
                ts=float(payload.get("ts") or time.time()),
                tool=name,
                truncated=truncated,
            )
        )
        self._trim_turns(agent)
        if agent.current_tool == name or name == "":
            agent.current_tool = None
        self._touch()

    def _tool_name_for_call(self, agent: SubagentProgress) -> Optional[str]:
        for turn in reversed(agent.turns):
            if turn.role == "tool_call":
                return turn.tool
            if turn.role == "tool_result":
                return None
        return None

    def _stream_aborted(self, agent_id: str) -> None:
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        self._close_stream_turn(agent, "assistant")
        self._close_stream_turn(agent, "reasoning")
        agent.current_tool = None
        self._touch()

    def _consume_response(self, agent_id: str, response: Any) -> None:
        kind = getattr(response, "type", "")
        if kind == "streaming_delta":
            text_kind, text = self._chain_text(response)
            self._stream_text(agent_id, text_kind, text)
        elif kind == "tool_call":
            payload = self._chain_json(response)
            if payload:
                self._stream_tool_call(agent_id, payload)
        elif kind in ("tool_call_result", "tool_direct_result"):
            payload = self._chain_json(response)
            if payload:
                self._stream_tool_result(agent_id, payload)
        elif kind == "agent_stats":
            agent = self._agents.get(agent_id)
            usage = None
            for comp in self._chain_components(response):
                blob = getattr(comp, "data", None)
                if isinstance(blob, dict):
                    usage = blob.get("token_usage")
                    break
            if agent is not None and isinstance(usage, dict):
                try:
                    agent.input_tokens = int(
                        usage.get("input_other", 0) or 0
                    ) + int(usage.get("input_cached", 0) or 0)
                    agent.output_tokens = int(usage.get("output", 0) or 0)
                except (TypeError, ValueError):
                    pass
        elif kind in ("aborted", "err"):
            self._stream_aborted(agent_id)

    def _mark_stream_dirty(self, agent: SubagentProgress) -> None:
        """Buffer text updates and flush on a timer instead of per token."""
        self._stream_dirty = True
        agent.updated_at = time.time()

    def _start_flush_loop(self) -> None:
        task = getattr(self, "_flush_task", None)
        if task is None or task.done():
            self._flush_task = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(STREAM_FLUSH_SECONDS)
                if self._stream_dirty:
                    self._stream_dirty = False
                    self._touch()
        except asyncio.CancelledError:
            pass

    def _trim_turns(self, agent: SubagentProgress) -> None:
        if len(agent.turns) <= MAX_TURNS_PER_AGENT:
            return
        del agent.turns[: len(agent.turns) - MAX_TURNS_PER_AGENT]
        for index, turn in enumerate(agent.turns):
            turn.seq = index

    def _note_file_access(self, agent: SubagentProgress, tool: str, args: dict) -> None:
        name = str(tool or "")
        path = args.get("path") or args.get("file_path") or args.get("filename")
        if not path:
            return
        path = str(path)
        if "write" in name or "edit" in name or "create" in name:
            if path not in agent.files_written:
                agent.files_written.append(path)
        elif "read" in name or "view" in name:
            if path not in agent.files_read:
                agent.files_read.append(path)

    @staticmethod
    def _safe_json(value: Any) -> Optional[dict]:
        if not isinstance(value, dict):
            return None
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return {"__unserializable__": str(value)[:500]}

    # ── change notification ─────────────────────────────────────────────

    def _touch(self) -> None:
        self._version += 1
        for waiter in list(self._waiters):
            waiter.set()

    async def _wait_for_change(self, timeout: float) -> None:
        waiter = asyncio.Event()
        self._waiters.add(waiter)
        try:
            await asyncio.wait_for(waiter.wait(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            pass
        finally:
            self._waiters.discard(waiter)

    def _snapshot(self, *, with_turns: bool = False, detail_id: Optional[str] = None) -> dict:
        agents = sorted(self._agents.values(), key=lambda a: a.started_at)
        payload = {
            "version": self._version,
            "agents": [a.to_dict(with_turns=with_turns) for a in agents],
            "server_time": time.time(),
        }
        # Ship the open conversation in the same round-trip. For a live view the
        # alternative is a second request per tick, which would double latency
        # and let the thread and the list disagree.
        if detail_id:
            agent = self._agents.get(detail_id)
            payload["detail"] = (
                {"id": detail_id, "turns": [t.to_dict() for t in agent.turns]}
                if agent is not None
                else {"id": detail_id, "turns": [], "missing": True}
            )
        return payload

    # ── eviction & persistence ──────────────────────────────────────────

    def _evict(self) -> None:
        if len(self._agents) <= MAX_AGENTS:
            return
        finished = sorted(
            (a for a in self._agents.values() if a.status in TERMINAL_STATUSES),
            key=lambda a: a.ended_at or a.updated_at,
        )
        for agent in finished:
            if len(self._agents) <= MAX_AGENTS:
                break
            self._agents.pop(agent.id, None)

    def _schedule_save(self) -> None:
        if not getattr(self, "_auto_save", True):
            return
        if self._save_pending:
            return
        self._save_pending = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._save_pending = False
            return
        loop.call_later(2.0, self._flush_save)

    def _flush_save(self) -> None:
        self._save_pending = False
        self._save_to_disk()

    def _save_to_disk(self) -> None:
        try:
            path: Path = self._store_path
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": time.time(),
                "agents": [a.to_storage() for a in self._agents.values() if not a.is_demo],
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.error("[subagent-viz] save failed: %s", exc)

    def _load_from_disk(self) -> None:
        try:
            path: Path = self._store_path
            if not path.exists():
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._load_error = str(exc)
            logger.error("[subagent-viz] could not read %s: %s", self._store_path, exc)
            return

        skipped = 0
        for raw in payload.get("agents", []) if isinstance(payload, dict) else []:
            # one malformed record must not cost us the whole history
            try:
                agent = SubagentProgress.from_storage(raw)
                if not agent.id:
                    continue
                if agent.status in (AgentStatus.RUNNING, AgentStatus.QUEUED):
                    agent.status = AgentStatus.INTERRUPTED
                    agent.error = agent.error or "AstrBot 重启导致运行中断"
                self._agents[agent.id] = agent
            except Exception as exc:
                skipped += 1
                logger.warning("[subagent-viz] skipped a stored record: %s", exc)
        if skipped:
            logger.warning("[subagent-viz] skipped %d unusable stored record(s)", skipped)

    # ── API ─────────────────────────────────────────────────────────────

    def _register_apis(self) -> None:
        # Routes are matched with re.fullmatch against the FULL path the host
        # builds, which already contains the plugin name: the webui prefixes
        # "/api/v1/plugins/extensions/{plugin_name}/" to the bare endpoint and
        # dashboard/api/plugins.py passes plugin_path straight to
        # _match_registered_web_api. The page still calls apiGet("agents") - it
        # never names the plugin itself. So PLUGIN_NAME must match metadata.yaml.
        reg = self.context.register_web_api
        reg(f"/{PLUGIN_NAME}/agents", self._api_agents, ["GET"], "Snapshot of all sub-agents")
        reg(f"/{PLUGIN_NAME}/events", self._api_events, ["GET"], "Long-poll for sub-agent changes")
        reg(f"/{PLUGIN_NAME}/agent", self._api_agent, ["GET"], "One sub-agent with its turns")
        reg(f"/{PLUGIN_NAME}/agent/kill", self._api_kill, ["POST"], "Stop a running sub-agent")
        reg(f"/{PLUGIN_NAME}/demo", self._api_demo, ["POST"], "Load demo data")
        reg(f"/{PLUGIN_NAME}/clear", self._api_clear, ["POST"], "Clear captured data")
        reg(f"/{PLUGIN_NAME}/export", self._api_export, ["GET"], "Export a sub-agent as JSON")

    async def _api_agents(self):
        detail_id = request.query.get("agent") or None
        if not self._agents and self._load_error:
            return json_response(
                {
                    "version": self._version,
                    "server_time": time.time(),
                    "agents": [],
                    "load_error": self._load_error,
                }
            )
        return json_response(self._snapshot(detail_id=detail_id))

    async def _api_events(self):
        raw = request.query.get("since", "0") or "0"
        try:
            since = int(raw)
        except (TypeError, ValueError):
            since = 0
        detail_id = request.query.get("agent") or None
        if since < self._version:
            return json_response(self._snapshot(detail_id=detail_id))
        deadline = time.monotonic() + LONG_POLL_SECONDS
        while self._version <= since:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await self._wait_for_change(min(remaining, 5.0))
        return json_response(self._snapshot(detail_id=detail_id))

    async def _api_agent(self):
        agent = self._agents.get(request.query.get("id", ""))
        if agent is None:
            return json_response({"error": "not found", "id": request.query.get("id", "")}, status_code=404)
        return json_response({"agent": agent.to_dict(with_turns=True)})

    async def _api_kill(self):
        body = await request.json({}) or {}
        agent_id = str(body.get("id", ""))
        agent = self._agents.get(agent_id)
        if agent is None:
            return json_response({"error": "not found"}, status_code=404)
        if agent.status in TERMINAL_STATUSES:
            return json_response({"error": "already finished", "status": agent.status.value})

        # Ask the capture hooks to unwind the sub-agent loop at its next
        # boundary. This is the only mechanism that works for a foreground
        # handoff, which shares the main agent's task.
        self._stop_requested.add(agent_id)

        # Background handoffs own their task, so those can be cancelled outright.
        task = self._tasks.get(agent_id)
        cancelled = False
        if task is not None and not task.done() and task not in self._main_tasks:
            task.cancel()
            cancelled = True

        self._finish_agent(agent_id, AgentStatus.INTERRUPTED, error=STOPPED_REASON)
        self._schedule_save()
        return json_response({"ok": True, "id": agent_id, "cancelled": cancelled})

    async def _api_demo(self):
        self._load_demo_data()
        return json_response({"ok": True, "count": len(self._agents)})

    async def _api_clear(self):
        body = await request.json({}) or {}
        scope = str(body.get("scope", "all"))
        if scope == "demo":
            for key in [k for k, v in self._agents.items() if v.is_demo]:
                self._agents.pop(key, None)
        else:
            self._agents.clear()
        self._touch()
        self._schedule_save()
        return json_response({"ok": True, "count": len(self._agents)})

    async def _api_export(self):
        agent = self._agents.get(request.query.get("id", ""))
        if agent is None:
            return json_response({"error": "not found"}, status_code=404)
        return json_response(
            {
                "id": agent.id,
                "name": agent.name,
                "goal": agent.goal,
                "status": agent.status.value,
                "model": agent.model,
                "started_at": agent.started_at,
                "ended_at": agent.ended_at,
                "tools": agent.tools,
                "input_tokens": agent.input_tokens,
                "output_tokens": agent.output_tokens,
                "turns": [t.to_dict() for t in agent.turns],
            },
            headers={"Content-Disposition": f'attachment; filename="{agent.id}.json"'},
        )

    # ── demo data ───────────────────────────────────────────────────────

    def _load_demo_data(self) -> None:
        for key in [k for k, v in self._agents.items() if v.is_demo]:
            self._agents.pop(key, None)
        now = time.time()

        def turn(role, text="", tool=None, args=None, is_error=False):
            return {
                "role": role,
                "text": text,
                "tool": tool,
                "args": args,
                "is_error": is_error,
            }

        specs = [
            {
                "id": "demo-root",
                "name": "planner",
                "parent_id": None,
                "goal": "Design a caching layer and delegate the two storage backends",
                "depth": 0,
                "started_at": now - 240,
                "ended_at": now - 40,
                "status": AgentStatus.COMPLETED,
                "tools": ["transfer_to_knowledge", "transfer_to_dev"],
                "turns": [
                    turn("user", "Design a caching layer and delegate the two storage backends"),
                    turn("reasoning", "Two independent backends - I can run them in parallel and review both."),
                    turn("tool_call", tool="transfer_to_knowledge",
                         args={"input": "Survey Redis vs Memcached eviction policies"}),
                    turn("tool_result", "Redis uses approximate LRU; Memcached uses strict LRU with a slab allocator.",
                         tool="transfer_to_knowledge"),
                    turn("tool_call", tool="transfer_to_dev",
                         args={"input": "Implement an LRU cache wrapper in Go"}),
                    turn("assistant", "Both backends are in. I went with Redis for the shared tier and an in-process LRU for the hot path."),
                ],
            },
            {
                "id": "demo-kb",
                "name": "knowledge",
                "parent_id": "demo-root",
                "goal": "Survey Redis vs Memcached eviction policies",
                "depth": 1,
                "started_at": now - 235,
                "ended_at": now - 150,
                "status": AgentStatus.COMPLETED,
                "tools": ["web_search", "read_file"],
                "turns": [
                    turn("user", "Survey Redis vs Memcached eviction policies"),
                    turn("tool_call", tool="web_search",
                         args={"query": "redis memcached eviction policy comparison"}),
                    turn("tool_result", "Redis: approximate LRU (sampling). Memcached: strict LRU over slab classes.",
                         tool="web_search"),
                    turn("tool_call", tool="read_file", args={"path": "docs/cache.md"}),
                    turn("assistant", "Redis approximates LRU by sampling 20 keys per eviction, which cuts CPU cost but can evict hot keys."),
                ],
            },
            {
                "id": "demo-dev",
                "name": "dev",
                "parent_id": "demo-root",
                "goal": "Implement an LRU cache wrapper in Go",
                "depth": 1,
                "started_at": now - 230,
                "ended_at": now - 45,
                "status": AgentStatus.FAILED,
                "error": "RuntimeError: go toolchain not available in sandbox",
                "tools": ["write_file", "run_command"],
                "turns": [
                    turn("user", "Implement an LRU cache wrapper in Go"),
                    turn("tool_call", tool="write_file", args={"path": "cache/lru.go"}),
                    turn("tool_result", "wrote 182 lines to cache/lru.go", tool="write_file"),
                    turn("tool_call", tool="run_command", args={"command": "go test ./cache/..."}),
                    turn("tool_result", "go: command not found", tool="run_command", is_error=True),
                    turn("error", "RuntimeError: go toolchain not available in sandbox"),
                ],
            },
            {
                "id": "demo-doc",
                "name": "doc",
                "parent_id": "demo-dev",
                "goal": "Document the LRU cache API",
                "depth": 2,
                "started_at": now - 120,
                "ended_at": None,
                "status": AgentStatus.RUNNING,
                "tools": ["read_file"],
                "current_tool": "read_file",
                "turns": [
                    turn("user", "Document the LRU cache API"),
                    turn("reasoning", "I should read the implementation before writing docs for it."),
                    turn("tool_call", tool="read_file", args={"path": "cache/lru.go"}),
                ],
            },
        ]

        for spec in specs:
            agent = SubagentProgress(
                id=spec["id"],
                name=spec["name"],
                parent_id=spec["parent_id"],
                goal=spec["goal"],
                model="demo/provider",
                status=spec["status"],
                depth=spec["depth"],
                started_at=spec["started_at"],
                ended_at=spec["ended_at"],
                updated_at=now,
                tools=list(spec["tools"]),
                tool_count=len(spec["tools"]),
                api_calls=len(spec["tools"]),
                current_tool=spec.get("current_tool"),
                error=spec.get("error"),
                input_tokens=1200 + spec["depth"] * 400,
                output_tokens=380 + spec["depth"] * 120,
                is_demo=True,
            )
            for index, item in enumerate(spec["turns"]):
                agent.turns.append(
                    Turn(
                        seq=index,
                        role=item["role"],
                        text=item["text"],
                        ts=spec["started_at"] + index * 8,
                        tool=item["tool"],
                        args=item["args"],
                        is_error=item["is_error"],
                    )
                )
            for turn_ in agent.turns:
                if turn_.role == "assistant" and turn_.text.strip():
                    agent.summary = turn_.text.strip()[:400]
                    break
            self._agents[agent.id] = agent
        self._touch()
