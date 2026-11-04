"""Language-model backends.

The agent only needs one method, ``chat(messages, tools) -> Reply``. Any model
that supports tool calling can be added by writing a small class with that
method. Two backends are included:

* ``OllamaBackend``  - a local model served by Ollama (http://localhost:11434)
* ``ScriptedBackend`` - replays a fixed list of replies; used in the tests so
  the agent loop can be checked without a model
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    from_text: bool = False  # True when parsed out of plain text, not native tool-call output


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None


class Backend(Protocol):
    name: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply: ...


_TOOL_JSON = re.compile(r"\{[^{}]*\"name\"\s*:\s*\"(?P<name>\w+)\"[^{}]*\"arguments\"\s*:\s*(?P<args>\{[^{}]*\})[^{}]*\}", re.S)


def parse_text_tool_calls(text: str) -> list[ToolCall]:
    """Small models sometimes write a tool call as JSON text instead of using the
    tool-call channel. Recover simple cases like
    ``{"name": "count_features", "arguments": {"layer": "plots"}}``."""
    calls = []
    for m in _TOOL_JSON.finditer(text or ""):
        try:
            calls.append(ToolCall(m.group("name"), json.loads(m.group("args")), from_text=True))
        except json.JSONDecodeError:
            continue
    return calls


class OllamaBackend:
    """Chat with a model served by a local Ollama instance."""

    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434",
                 temperature: float = 0.2, seed: int | None = None, timeout: float = 300.0,
                 num_ctx: int = 8192):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.name = f"ollama:{model}"

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.host + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.host}. Is it running? ({exc.reason})") from exc

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        options: dict[str, Any] = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        if self.seed is not None:
            options["seed"] = self.seed
        data = self._post("/api/chat", {"model": self.model, "messages": messages, "tools": tools,
                                        "stream": False, "options": options})
        msg = data.get("message", {})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_unparsed": args}
            calls.append(ToolCall(fn.get("name", ""), args))
        content = msg.get("content") or ""
        if not calls:
            calls = parse_text_tool_calls(content)
        return Reply(content=content, tool_calls=calls, raw=data)


class ScriptedBackend:
    """Return pre-written replies in order (for tests and offline demos)."""

    def __init__(self, replies: list[Reply]):
        self.replies = list(replies)
        self.name = "scripted"

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        if not self.replies:
            return Reply(content="ANSWER: none (script exhausted)")
        return self.replies.pop(0)
