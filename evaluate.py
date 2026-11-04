"""Run the benchmark and write the results tables.

Examples
--------
    python evaluate.py                                # qwen2.5:3b, 3 runs, checks on (warn)
    python evaluate.py --modes off warn               # compare with checks hidden from the model
    python evaluate.py --questions Q04 Q06 --runs 1   # quick try
    python evaluate.py --report-only                  # rebuild tables from results/runs.csv

Outputs (in results/):
    runs.csv        one row per run
    summary.md      tables used in the README
Each run's full step-by-step trace is saved in traces/.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from geo_agent.agent import Agent, RunResult
from geo_agent.benchmark import QUESTIONS, is_correct, parse_number, reference_answers
from geo_agent.llm import OllamaBackend

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TRACES = ROOT / "traces" / "eval"

FIELDS = ["mode", "qid", "run", "model", "finished", "correct", "answer", "reference", "failure",
          "grounded", "n_tool_calls", "n_tool_errors", "checks_fired", "geo_crs_unfixed", "text_tool_calls",
          "stop_reason", "elapsed_s", "trace"]


def _numbers_in(obj: Any) -> list[float]:
    """All numbers appearing anywhere in a tool result."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", obj)]
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += _numbers_in(k) + _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers_in(v)
    return out


def grounded(answer: str | None, steps: list[dict[str, Any]], kind: str) -> bool | None:
    """Can the answer be found in a tool result? Numbers may be rounded (0.5% or 2 decimals)."""
    if answer is None:
        return None
    results = [s.get("result") for s in steps if s["type"] == "tool" and "result" in s]
    if kind == "name":
        blob = json.dumps(results, default=str).lower()
        token = re.sub(r"[^\w]", "", answer.split()[0].lower()) if answer.split() else ""
        return bool(token) and token in blob
    val = parse_number(answer)
    if val is None:
        return None
    for n in _numbers_in(results):
        if abs(n - val) <= max(0.005 * abs(n), 0.006):
            return True
    # Counting the rows of a per-feature result is also grounded.
    for r in results:
        for key in ("values",):
            if isinstance(r, dict) and isinstance(r.get(key), dict) and len(r[key]) == val:
                return True
    return False


def classify(res: RunResult, correct: bool | None, is_grounded: bool | None) -> str:
    """One failure label per run (empty string when the answer is correct)."""
    if correct:
        return ""
    if res.stop_reason == "llm_error":
        return "llm_error"
    if res.stop_reason == "max_steps":
        return "no_answer_max_steps"
    if correct is None:
        return "unparseable_answer"
    tool_steps = [s for s in res.steps if s["type"] == "tool"]
    executed = [s for s in tool_steps if "result" in s]
    codes = {f["code"] for s in executed for f in s.get("checks", [])}
    if "GEOGRAPHIC_CRS" in codes:
        return "measured_in_degrees"
    if "CRS_MISMATCH" in codes:
        return "crs_mismatch"
    if is_grounded is False:
        return "answer_not_in_tool_output"
    if any("error" in s for s in tool_steps) and not executed:
        return "tool_errors_only"
    return "wrong_workflow"


def run_one(model: str, mode: str, q, run: int, ref: Any, max_steps: int, temperature: float) -> dict[str, Any]:
    backend = OllamaBackend(model=model, temperature=temperature, seed=1000 + run)
    agent = Agent(backend, check_mode=mode, max_steps=max_steps, trace_dir=TRACES)
    run_id = f"{mode}_{q.qid}_r{run}"
    res = agent.run(q.text, run_id=run_id)
    corr = is_correct(q, res.answer_text, ref)
    grd = grounded(res.answer_text, res.steps, q.kind)
    tool_steps = [s for s in res.steps if s["type"] == "tool"]
    executed = [s for s in tool_steps if "result" in s]
    return {
        "mode": mode, "qid": q.qid, "run": run, "model": model,
        "finished": res.finished, "correct": bool(corr), "answer": res.answer_text,
        "reference": round(ref, 4) if isinstance(ref, float) else ref,
        "failure": classify(res, corr, grd), "grounded": grd,
        "n_tool_calls": len(tool_steps),
        "n_tool_errors": sum(1 for s in tool_steps if "error" in s),
        "checks_fired": sum(len(s.get("checks", [])) for s in tool_steps),
        "geo_crs_unfixed": any(f["code"] == "GEOGRAPHIC_CRS" for s in executed for f in s.get("checks", [])),
        "text_tool_calls": sum(1 for s in tool_steps if s.get("from_text")),
        "stop_reason": res.stop_reason, "elapsed_s": res.elapsed_s,
        "trace": Path(res.trace_path).relative_to(ROOT).as_posix() if res.trace_path else "",
    }


def _b(x: str) -> bool:
    return str(x).lower() == "true"


def write_summary(rows: list[dict[str, Any]], path: Path) -> str:
    by_mode: dict[str, list] = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    models = sorted({r["model"] for r in rows})
    lines = [f"Model: {', '.join(models)}. Runs per question: "
             f"{max(int(r['run']) for r in rows) + 1}. Generated by `evaluate.py` on {time.strftime('%Y-%m-%d')}.", ""]

    lines += ["### Overall", "",
              "| Check mode | Runs | Finished | Correct | Answer found in tool output | Ran a measurement in degrees | Mean tool calls | Mean time (s) |",
              "|---|---|---|---|---|---|---|---|"]
    for mode, rs in by_mode.items():
        n = len(rs)
        fin = sum(_b(r["finished"]) for r in rs)
        cor = sum(_b(r["correct"]) for r in rs)
        grd = sum(_b(r["grounded"]) for r in rs)
        deg = sum(_b(r["geo_crs_unfixed"]) for r in rs)
        tc = sum(int(r["n_tool_calls"]) for r in rs) / n
        t = sum(float(r["elapsed_s"]) for r in rs) / n
        lines.append(f"| {mode} | {n} | {fin} ({fin / n:.0%}) | {cor} ({cor / n:.0%}) | {grd} ({grd / n:.0%}) "
                     f"| {deg} | {tc:.1f} | {t:.0f} |")

    modes = list(by_mode)
    lines += ["", "### Per question (correct runs / total)", "",
              "| Question | Tests | Reference | " + " | ".join(modes) + " |",
              "|---|---|---|" + "---|" * len(modes)]
    qinfo = {q.qid: q for q in QUESTIONS}
    for qid in sorted({r["qid"] for r in rows}):
        q = qinfo[qid]
        ref = next(r["reference"] for r in rows if r["qid"] == qid)
        cells = []
        for m in modes:
            rs = [r for r in by_mode[m] if r["qid"] == qid]
            cells.append(f"{sum(_b(r['correct']) for r in rs)}/{len(rs)}")
        lines.append(f"| {qid}: {q.text} | {q.tests} | {ref} | " + " | ".join(cells) + " |")

    lines += ["", "### Failure types (runs with a wrong or missing answer)", "",
              "| Failure type | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    counts = {m: Counter(r["failure"] for r in by_mode[m] if r["failure"]) for m in modes}
    for ft in sorted(set().union(*counts.values())):
        lines.append(f"| {ft} | " + " | ".join(str(counts[m].get(ft, 0)) for m in modes) + " |")
    if not any(counts.values()):
        lines.append("| (none) | " + " | ".join("0" for _ in modes) + " |")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["warn"], choices=["off", "warn", "block"])
    ap.add_argument("--questions", nargs="*", help="question ids, e.g. Q01 Q04 (default: all)")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--report-only", action="store_true", help="only rebuild summary.md from runs.csv")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / "runs.csv"
    if args.report_only:
        with csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        print(write_summary(rows, RESULTS / "summary.md"))
        return

    refs = reference_answers()
    qs = [q for q in QUESTIONS if not args.questions or q.qid in args.questions]
    rows: list[dict[str, Any]] = []
    total = len(qs) * args.runs * len(args.modes)
    t0 = time.time()
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for mode in args.modes:
            for q in qs:
                for run in range(args.runs):
                    row = run_one(args.model, mode, q, run, refs[q.qid], args.max_steps, args.temperature)
                    rows.append(row)
                    w.writerow(row)
                    fh.flush()
                    mark = "ok " if row["correct"] else "BAD"
                    print(f"[{len(rows)}/{total}] {mode:5s} {q.qid} r{run} {mark} answer={row['answer']!r} "
                          f"ref={row['reference']} {row['failure']} ({row['elapsed_s']:.0f}s, "
                          f"{time.time() - t0:.0f}s total)", flush=True)
    print()
    print(write_summary([{k: str(v) for k, v in r.items()} for r in rows], RESULTS / "summary.md"))


if __name__ == "__main__":
    main()
