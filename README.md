# geo-llm-agent-demo

A small test of a question I want to work on: when a language model plans and runs a GIS
workflow by itself, how often is the result right, and can simple checks catch the mistakes?

The agent answers plain-English spatial questions ("What is the total area of rice plots in
hectares?") by calling GeoPandas tools through a local model served by
[Ollama](https://ollama.com). After every tool call, a set of checks looks for common
geospatial errors. A benchmark of 12 questions with known answers measures how often the
agent gets things right, and in what ways it fails.

This is a first step, about a day of work. It is not a RAG system or a multi-agent system,
and the data are synthetic.

## What is in the repository

| Path | What it does |
|---|---|
| `geo_agent/data.py` | Generates a synthetic study area near Gazipur, Bangladesh: 24 farm plots, 3 villages, roads, a canal, 40 soil samples, a 10 m NDVI raster. Fixed seed, so the answers never change. |
| `geo_agent/tools.py` | 15 tools (plain Python functions, each with a JSON schema): list/load/inspect layers, reproject, fix geometries, filter, buffer, spatial join, clip, measure area/length, count, summarise, zonal statistics, save, plot. |
| `geo_agent/checks.py` | Checks before and after each tool call: geographic CRS used for measurement, CRS mismatch between layers, invalid geometries, empty results, implausible values. |
| `geo_agent/agent.py` | ReAct-style loop: the model calls a tool, reads the result (with any check findings), decides the next step, and finally answers. Every step is saved to a JSON trace. |
| `geo_agent/llm.py` | Model backends. `OllamaBackend` for a local model, `ScriptedBackend` for tests. Adding another backend means writing one `chat()` method. |
| `geo_agent/benchmark.py` | 12 questions with reference answers computed directly in GeoPandas (always in UTM 46N, after repairing geometries), separately from the agent's tools. |
| `evaluate.py` | Runs the agent on the benchmark several times, scores each run and writes `results/runs.csv` and `results/summary.md`. |
| `ask.py` | Ask one question from the command line and watch the steps. |
| `tests/test_demo.py` | Tests that need no model, including one hand-written tool workflow per benchmark question showing that the tools can reach every reference answer. |

### The traps in the data

The data are built so that a careless workflow gives a wrong number:

- Plots, roads and soil samples are stored in longitude/latitude (EPSG:4326, degrees);
  villages, the canal and the NDVI raster are in UTM zone 46N (EPSG:32646, metres). Measuring
  area in degrees gives about 5 × 10⁻¹⁰ "ha" instead of about 6 ha. Joining a degree layer with a metre layer
  matches nothing.
- One plot (P24) is a self-intersecting "bow-tie" polygon. Its area is 0 until it is
  repaired, which makes the Kaliakoir village total (Q08) about 6% too low.
- A 50 m buffer applied to a layer in degrees covers the whole study area, so every plot
  appears "near the canal".

The tools do not protect against these traps. They behave like plain GeoPandas code. Catching the
mistakes is the job of the checks, so the two can be evaluated separately.

## How to run (Windows, PowerShell)

Requirements: Python 3.11 and Ollama for Windows.

```powershell
cd E:\geo-llm-agent-demo
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m geo_agent.data        # writes the synthetic layers to data/ (already included)
python -m pytest -q             # 24 tests, no model needed
```

Get a small model with tool calling (about 1.9 GB; runs on CPU with 8 GB RAM):

```powershell
ollama pull qwen2.5:3b
```

Ask one question:

```powershell
python ask.py "What is the total area of rice plots in hectares?"
python ask.py "How many farm plots lie within 50 metres of the canal?" --checks off
```

Run the evaluation (12 questions × 3 runs; takes a while on CPU):

```powershell
python evaluate.py                      # checks shown to the model ("warn")
python evaluate.py --modes off warn     # compare with checks hidden from the model
python evaluate.py --model llama3.2:3b  # any other Ollama model with tool support
```

On macOS/Linux, use `source .venv/bin/activate`; the other commands are the same.

### Check modes

| Mode | What the model sees |
|---|---|
| `off` | Nothing. The checks still run and are written to the trace, so failures can be classified, but the model gets plain tool results. This is the baseline. |
| `warn` | Check findings are added to the tool result, e.g. "Layer 'plots' is in a geographic CRS ... Reproject first". |
| `block` | Like `warn`, but a measurement in degrees or a join across CRSs is refused until the model fixes it. |

## What a step looks like in a trace

Each run writes `traces/.../<run_id>.json` with the model's messages and every tool call.
This is the tool step written when a model measures area directly on the lon/lat layer
(taken from the scripted test; the number is what plain GeoPandas returns):

```json
{
  "type": "tool",
  "tool": "measure",
  "args": {"layer": "plots", "quantity": "area", "unit": "ha"},
  "blocked": false,
  "result": {"layer": "plots", "quantity": "area", "unit": "ha", "crs": "EPSG:4326",
             "n_features": 24, "value": 5.03e-10},
  "checks": [
    {"code": "GEOGRAPHIC_CRS", "severity": "error",
     "message": "Layer 'plots' is in a geographic CRS (EPSG:4326, units: degrees). area would be computed in degrees, not metres. Reproject to a projected CRS first (e.g. EPSG:32646, UTM zone 46N)."},
    {"code": "INVALID_GEOMETRY", "severity": "warning",
     "message": "Layer 'plots' has 1 invalid geometry; areas and overlays may be wrong. Consider fix_geometries."},
    {"code": "UNIT_SANITY", "severity": "warning",
     "message": "Area 5.03e-10 ha is implausibly small; was it measured in degrees?"}
  ]
}
```

## How a run is scored

For each run, `evaluate.py` records:

- **Finished**: the model gave a final `ANSWER:` line within 12 steps.
- **Correct**: counts and names must match exactly; areas, lengths and NDVI within 2%.
- **Answer found in tool output**: the number (or name) in the answer appears in some
  tool result. An answer that is not there was made up or calculated by the model itself,
  which is worth knowing even when it happens to be right.
- **Failure type**, one per wrong run, decided from the trace in this order:
  `llm_error`, `no_answer_max_steps`, `unparseable_answer`, `measured_in_degrees`
  (an area/length/buffer was actually computed on a degree layer), `crs_mismatch`,
  `answer_not_in_tool_output`, `tool_errors_only`, `wrong_workflow` (everything else, e.g.
  wrong filter or wrong predicate).

## Results

**Not run yet.** The code and the scoring have been tested with the scripted backend only
(`python -m pytest`). The model evaluation has not been run on this machine yet, so there are
no results to report. After `python evaluate.py`, the tables appear in
`results/summary.md` and should be copied here unchanged, including the failures.

## Limitations

- **Synthetic data, small benchmark.** 12 questions on one made-up study area. The data are
  built to contain known traps, so the results say more about those traps than about GIS
  work in general.
- **One small model.** A 3B model on CPU makes mistakes a larger model may not, and the
  reverse is also possible. Results depend on the model, the prompt and the temperature (0.2).
- **The prompt helps.** The system prompt already says that areas need a projected CRS and
  names EPSG:32646. Without that hint, results would probably be worse. A real user would
  not always give it.
- **The checks are hand-written rules** for this study area (e.g. "an area above 1,000 ha
  is implausible"). They catch the traps I built in. They will miss errors I did not think of,
  such as a wrong spatial predicate or a wrong attribute filter.
- **Scoring is automatic and simple.** The first number in the answer is compared with the
  reference. A correct number with the wrong unit, or an answer that states two numbers,
  can be scored wrongly. The traces are kept so that any run can be checked by hand.
- **Reference answers involve choices.** For example, the bow-tie plot is repaired with
  `make_valid` before measuring, and Q11 averages the per-plot means. Other reasonable
  choices would give slightly different references.

## How this relates to evaluating AI-generated geospatial analyses

An LLM can produce a GIS workflow that runs without errors and still gives a wrong answer.
Measuring in degrees is the classic case: GeoPandas only prints a warning. So "the code ran"
is not the same as "the analysis is right". This demo tries three simple ways to look at
reliability:

1. **Ground truth.** Compare the final answer with a reference computed independently.
2. **Process checks.** Look at *how* the answer was reached (CRS, geometry validity, empty
   intermediate results), not only at the final number. The trace makes every step
   inspectable.
3. **Grounding.** Check that the reported number actually came from a tool, rather than from
   the model's own arithmetic or memory.

Comparing `off` and `warn` asks a narrower question: do the check messages help the model
correct itself, or does it ignore them?

Possible next steps: more varied and realistic data (e.g. real UAV plot layers), questions
with more than one valid workflow, a larger model and a second model family for comparison,
checks that compare an LLM workflow with a reference workflow step by step, and retrieval of
GIS documentation for the agent.

## Author

Md. Khalid Hasan Milu ([GitHub](https://github.com/milukhalidhasan),
ORCID [0009-0008-5927-5814](https://orcid.org/0009-0008-5927-5814)).
Written with help from an AI coding assistant; the design, the data traps and the
evaluation choices are explained above so they can be checked.
