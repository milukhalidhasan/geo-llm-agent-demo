"""A small ReAct-style agent loop.

The model sees the question and the tool list. Each turn it either calls one or
more tools or gives its final answer. After every tool call the checks in
``checks.py`` run, and (depending on the mode) their findings are shown to the
model with the tool result. Every step is written to a JSON trace.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import checks
from .llm import Backend
from .tools import TOOL_SCHEMAS, ToolError, Workspace, call_tool, to_json

TRACE_DIR = Path(__file__).resolve().parent.parent / "traces"

SYSTEM_PROMPT = """You are a GIS analyst. Answer the user's question about the data in the workspace \
by calling the tools. Work step by step: inspect layers before using them, then call one tool at a time \
and read its result before deciding the next step.

Workspace layers: plots (farm plots, columns plot_id, crop, n_rate in kg N/ha), villages (village \
boundaries, column village), roads (road lines, columns road_id, type), canal (irrigation canal line), \
samples (soil sampling points, columns sample_id, soc_pct). Raster: ndvi (NDVI, 10 m).

Rules:
- Layers may be stored in different coordinate reference systems. Distances and areas must be computed in \
a projected CRS in metres; EPSG:32646 (UTM zone 46N) suits this area.
- Tool results may include "checks" with warnings or errors. Read them and fix the problem before going on.
- Do not guess numbers. Every number in your answer must come from a tool result.
- When you have the answer, reply with one final line in the form
ANSWER: <value> <unit>
using a single number (or a single name for "which" questions)."""

ANSWER_RE = re.compile(r"ANSWER\s*:\s*(?P<ans>.+)", re.I)


@dataclass
class RunResult:
    question: str
    answer_text: str | None
    finished: bool
    stop_reason: str  # "answer", "max_steps", "llm_error"
    steps: list[dict[str, Any]] = field(default_factory=list)
    trace_path: str | None = None
    elapsed_s: float = 0.0

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s["type"] == "tool")


def extract_answer(text: str) -> str | None:
    """Return the text after the last 'ANSWER:' marker, or None."""
    matches = ANSWER_RE.findall(text or "")
    return matches[-1].strip().strip("*` ") if matches else None


class Agent:
    def __init__(self, backend: Backend, ws: Workspace | None = None, check_mode: str = "warn",
                 max_steps: int = 12, trace_dir: Path | None = TRACE_DIR, verbose: bool = False):
        if check_mode not in checks.MODES:
            raise ValueError(f"check_mode must be one of {checks.MODES}")
        self.backend = backend
        self.ws = ws or Workspace.default()
        self.check_mode = check_mode
        self.max_steps = max_steps
        self.trace_dir = trace_dir
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _run_tool(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one tool with checks. Returns (observation for the model, trace record)."""
        pre = checks.pre_checks(self.ws, name, args)
        blocked = self.check_mode == "block" and any(f.severity == "error" for f in pre)
        record: dict[str, Any] = {"type": "tool", "tool": name, "args": args, "blocked": blocked}
        post: list[checks.Finding] = []
        if blocked:
            obs: dict[str, Any] = {"status": "blocked", "reason": "A check failed; fix it and try again."}
        else:
            try:
                result = call_tool(self.ws, name, args)
                obs = {"status": "ok", "result": result}
                record["result"] = result
                post = checks.post_checks(self.ws, name, args, result)
            except ToolError as exc:
                obs = {"status": "error", "error": str(exc)}
                record["error"] = str(exc)
        findings = checks.findings_to_dicts(pre + post)
        record["checks"] = findings
        if findings and self.check_mode != "off":
            obs["checks"] = findings
        return obs, record

    def run(self, question: str, run_id: str | None = None) -> RunResult:
        run_id = run_id or uuid.uuid4().hex[:8]
        t0 = time.time()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        steps: list[dict[str, Any]] = []
        answer, stop = None, "max_steps"
        for i in range(self.max_steps):
            try:
                t1 = time.time()
                reply = self.backend.chat(messages, TOOL_SCHEMAS)
            except Exception as exc:  # network errors, model not found ...
                steps.append({"type": "llm_error", "step": i, "error": str(exc)})
                stop = "llm_error"
                break
            steps.append({"type": "llm", "step": i, "content": reply.content,
                          "tool_calls": [{"name": c.name, "arguments": c.arguments, "from_text": c.from_text}
                                         for c in reply.tool_calls],
                          "text_fallback": bool(reply.raw and reply.raw.get("text_fallback")),
                          "seconds": round(time.time() - t1, 2)})
            self._log(f"[{i}] model: {reply.content.strip()[:200]!r} calls={[c.name for c in reply.tool_calls]}")
            if not reply.tool_calls:
                answer = extract_answer(reply.content)
                if answer is not None:
                    stop = "answer"
                    break
                # No tool call and no answer: remind the model of the format once per turn.
                messages.append({"role": "assistant", "content": reply.content})
                messages.append({"role": "user", "content": "Your reply had no tool call and no ANSWER line. "
                                                            "Call a tool now, or give the final line "
                                                            "'ANSWER: <value> <unit>'."})
                continue
            messages.append({"role": "assistant", "content": reply.content,
                             "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}}
                                            for c in reply.tool_calls]})
            for call in reply.tool_calls:
                obs, record = self._run_tool(call.name, call.arguments)
                record["step"] = i
                record["from_text"] = call.from_text
                steps.append(record)
                self._log(f"    -> {call.name}({call.arguments}) {to_json(obs)[:300]}")
                messages.append({"role": "tool", "tool_name": call.name, "content": to_json(obs)})
        res = RunResult(question=question, answer_text=answer, finished=answer is not None, stop_reason=stop,
                        steps=steps, elapsed_s=round(time.time() - t0, 2))
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.trace_dir / f"{run_id}.json"
            path.write_text(json.dumps({
                "run_id": run_id, "backend": self.backend.name, "check_mode": self.check_mode,
                "question": question, "answer": answer, "stop_reason": stop,
                "elapsed_s": res.elapsed_s, "steps": steps,
            }, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
            res.trace_path = str(path)
        return res
