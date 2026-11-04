"""Tests that need no language model.

* Every benchmark question can be solved with the agent's tools (so a wrong
  agent answer is the agent's mistake, not a missing tool).
* The checks fire on the traps they are meant to catch.
* The agent loop runs end to end with a scripted backend and writes a trace.

Run:  python -m pytest -q
"""
from __future__ import annotations

import json

import pytest

from geo_agent import checks
from geo_agent.agent import Agent, extract_answer
from geo_agent.benchmark import QUESTIONS, is_correct, reference_answers
from geo_agent.llm import Reply, ScriptedBackend, ToolCall, parse_text_tool_calls
from geo_agent.tools import ToolError, Workspace, call_tool

REFS = reference_answers()
Q = {q.qid: q for q in QUESTIONS}


@pytest.fixture()
def ws(tmp_path):
    w = Workspace.default()
    w.output_dir = tmp_path
    return w


def T(ws, name, **args):
    return call_tool(ws, name, args)


def _utm(ws, layer, out=None):
    return T(ws, "reproject", layer=layer, epsg=32646, out_name=out or f"{layer}_utm")


# One correct tool workflow per question. Each returns the answer as text.
SOLUTIONS = {
    "Q01": lambda ws: str(T(ws, "count_features", layer="plots")["count"]),
    "Q02": lambda ws: T(ws, "inspect_layer", layer="plots")["crs"],
    "Q03": lambda ws: str(T(ws, "inspect_layer", layer="plots")["n_invalid_geometries"]),
    "Q04": lambda ws: (_utm(ws, "plots"),
                       T(ws, "filter_features", layer="plots_utm", column="crop", op="==", value="rice", out_name="rice"),
                       str(T(ws, "measure", layer="rice", quantity="area", unit="ha")["value"]))[-1],
    "Q05": lambda ws: (_utm(ws, "roads"), str(T(ws, "measure", layer="roads_utm", quantity="length", unit="km")["value"]))[-1],
    "Q06": lambda ws: (_utm(ws, "plots"),
                       T(ws, "buffer", layer="canal", distance=50, out_name="canal50"),
                       str(T(ws, "spatial_join", left="plots_utm", right="canal50", predicate="intersects",
                             out_name="near")["n_features"]))[-1],
    "Q07": lambda ws: (_utm(ws, "plots"),
                       T(ws, "spatial_join", left="plots_utm", right="villages", predicate="within", out_name="pv"),
                       next(iter(T(ws, "summarise", layer="pv", column="plot_id", stat="count",
                                   group_by="village")["values"])))[-1],
    "Q08": lambda ws: (T(ws, "fix_geometries", layer="plots", out_name="plots_ok"), _utm(ws, "plots_ok", "plots_utm"),
                       T(ws, "filter_features", layer="villages", column="village", op="==", value="Kaliakoir", out_name="k"),
                       T(ws, "spatial_join", left="plots_utm", right="k", predicate="within", out_name="pk"),
                       str(T(ws, "measure", layer="pk", quantity="area", unit="ha")["value"]))[-1],
    "Q09": lambda ws: (T(ws, "filter_features", layer="plots", column="crop", op="==", value="wheat", out_name="w"),
                       str(T(ws, "spatial_join", left="samples", right="w", predicate="within",
                             out_name="sw")["n_features"]))[-1],
    "Q10": lambda ws: (_utm(ws, "plots"),
                       T(ws, "filter_features", layer="plots_utm", column="n_rate", op=">=", value=120, out_name="hi"),
                       str(T(ws, "measure", layer="hi", quantity="area", unit="ha", aggregate="mean")["value"]))[-1],
    "Q11": lambda ws: (T(ws, "filter_features", layer="plots", column="crop", op="==", value="wheat", out_name="w"),
                       T(ws, "zonal_stats", layer="w", raster="ndvi", stat="mean", out_name="wz"),
                       str(T(ws, "summarise", layer="wz", column="ndvi_mean", stat="mean")["value"]))[-1],
    "Q12": lambda ws: (T(ws, "zonal_stats", layer="plots", raster="ndvi", stat="mean", out_name="z"),
                       (lambda v: max(v, key=v.get))(T(ws, "zonal_stats", layer="plots", raster="ndvi", stat="mean",
                                                       out_name="z")["values"]))[-1],
}


def test_every_question_has_a_solution():
    assert set(SOLUTIONS) == set(Q)


@pytest.mark.parametrize("qid", sorted(SOLUTIONS))
def test_question_is_solvable_with_tools(ws, qid):
    answer = SOLUTIONS[qid](ws)
    assert is_correct(Q[qid], answer, REFS[qid]), f"{qid}: tools gave {answer}, reference {REFS[qid]}"


def test_area_in_degrees_is_wrong_and_flagged(ws):
    args = {"layer": "plots", "quantity": "area", "unit": "ha"}
    found = checks.pre_checks(ws, "measure", args)
    assert any(f.code == "GEOGRAPHIC_CRS" for f in found)
    res = T(ws, "measure", **args)
    assert not is_correct(Q["Q04"], str(res["value"]), REFS["Q04"])
    assert any(f.code == "UNIT_SANITY" for f in checks.post_checks(ws, "measure", args, res))


def test_crs_mismatch_flagged(ws):
    found = checks.pre_checks(ws, "spatial_join", {"left": "plots", "right": "villages", "predicate": "within"})
    assert [f.code for f in found].count("CRS_MISMATCH") == 1
    res = T(ws, "spatial_join", left="plots", right="villages", predicate="within", out_name="x")
    assert res["n_features"] == 0  # no overlap: degrees vs metres
    assert any(f.code == "EMPTY_RESULT" for f in checks.post_checks(ws, "spatial_join", {}, res))


def test_invalid_geometry_flagged(ws):
    found = checks.pre_checks(ws, "measure", {"layer": "plots", "quantity": "area", "unit": "ha"})
    assert any(f.code == "INVALID_GEOMETRY" for f in found)


def test_unknown_layer_and_tool_raise(ws):
    with pytest.raises(ToolError):
        T(ws, "count_features", layer="fields")
    with pytest.raises(ToolError):
        call_tool(ws, "delete_everything", {})
    with pytest.raises(ToolError):
        T(ws, "count_features", layer="plots", colour="red")


def test_extract_answer():
    assert extract_answer("I measured it.\nANSWER: 2.98 ha") == "2.98 ha"
    assert extract_answer("**ANSWER:** 12") == "12"
    assert extract_answer("no marker") is None


def test_parse_text_tool_calls():
    calls = parse_text_tool_calls('<tool_call>{"name": "count_features", "arguments": {"layer": "plots"}}</tool_call>')
    assert calls and calls[0].name == "count_features" and calls[0].arguments == {"layer": "plots"}


def _degrees_script():
    return [
        Reply(tool_calls=[ToolCall("measure", {"layer": "plots", "quantity": "area", "unit": "ha"})]),
        Reply(content="ANSWER: 5e-10 ha"),
    ]


@pytest.mark.parametrize("mode,expect_blocked,expect_shown", [("off", False, False), ("warn", False, True),
                                                              ("block", True, True)])
def test_check_modes(tmp_path, mode, expect_blocked, expect_shown):
    backend = ScriptedBackend(_degrees_script())
    agent = Agent(backend, check_mode=mode, trace_dir=tmp_path)
    res = agent.run("What is the total plot area in hectares?", run_id="t")
    step = next(s for s in res.steps if s["type"] == "tool")
    assert step["blocked"] is expect_blocked
    assert any(c["code"] == "GEOGRAPHIC_CRS" for c in step["checks"])  # always logged
    trace = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert trace["check_mode"] == mode and trace["answer"] == "5e-10 ha"


def test_agent_end_to_end_with_script(tmp_path):
    ref = REFS["Q04"]
    script = [
        Reply(tool_calls=[ToolCall("inspect_layer", {"layer": "plots"})]),
        Reply(tool_calls=[ToolCall("reproject", {"layer": "plots", "epsg": 32646, "out_name": "p"})]),
        Reply(tool_calls=[ToolCall("filter_features", {"layer": "p", "column": "crop", "op": "==",
                                                       "value": "rice", "out_name": "r"})]),
        Reply(tool_calls=[ToolCall("measure", {"layer": "r", "quantity": "area", "unit": "ha"})]),
        Reply(content=f"The rice plots cover {ref:.2f} ha.\nANSWER: {ref:.2f} ha"),
    ]
    agent = Agent(ScriptedBackend(script), trace_dir=tmp_path)
    res = agent.run(Q["Q04"].text, run_id="e2e")
    assert res.stop_reason == "answer" and res.n_tool_calls == 4
    assert is_correct(Q["Q04"], res.answer_text, ref)
    measure_step = [s for s in res.steps if s.get("tool") == "measure"][0]
    assert not any(c["code"] == "GEOGRAPHIC_CRS" for c in measure_step["checks"])


def test_grounding_check():
    from evaluate import grounded

    steps = [{"type": "tool", "result": {"value": 2.98154}}]
    assert grounded("2.98 ha", steps, "number") is True
    assert grounded("3.40 ha", steps, "number") is False


def test_ollama_empty_reply_uses_text_fallback():
    """An empty reply (a tool call Ollama could not parse) is retried in plain text."""
    from geo_agent.llm import OllamaBackend

    class FakeOllama(OllamaBackend):
        def __init__(self):
            super().__init__()
            self.payloads = []

        def _post(self, path, payload):
            self.payloads.append(payload)
            if len(self.payloads) == 1:
                return {"message": {"content": ""}, "eval_count": 41}
            return {"message": {"content": '{"name": "reproject_layer", "arguments": {"layer": "plots"}}'}}

    b = FakeOllama()
    reply = b.chat([{"role": "user", "content": "q"}], [{"function": {"name": "reproject"}}])
    assert "tools" in b.payloads[0] and "tools" not in b.payloads[1]
    assert reply.tool_calls[0].name == "reproject_layer" and reply.tool_calls[0].from_text
    assert reply.raw["text_fallback"] is True
