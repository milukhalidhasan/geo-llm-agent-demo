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
python -m pytest -q             # 25 tests, no model needed
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
python evaluate.py --modes off warn --resume   # continue an interrupted evaluation
```

On this laptop (CPU only) the 72-run evaluation took about 2.5 hours.

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
  `llm_error`, `no_answer_max_steps`, `gave_up` (answered "None"), `unparseable_answer`,
  `measured_in_degrees`
  (an area/length/buffer was actually computed on a degree layer), `crs_mismatch`,
  `answer_not_in_tool_output`, `tool_errors_only`, `wrong_workflow` (everything else, e.g.
  wrong filter or wrong predicate).

## Results

Run on 4 November 2026 on a laptop (Windows 11, 7.6 GB RAM, CPU only, no GPU) with
`python evaluate.py --modes off warn`: `qwen2.5:3b` through Ollama 0.40.14, temperature 0.2,
12 questions × 3 runs × 2 check modes = 72 runs. The run was interrupted after 59 runs and
finished with `--resume`; each run is independent, so this does not change the results.
The tables below are copied unchanged from `results/summary.md`. Every run's trace is in
`traces/eval/`.

### Overall

| Check mode | Runs | Finished | Correct | Answer found in tool output | Ran a measurement in degrees | Mean tool calls | Mean time (s) |
|---|---|---|---|---|---|---|---|
| off | 36 | 33 (92%) | 10 (28%) | 31 (86%) | 6 | 3.8 | 138 |
| warn | 36 | 29 (81%) | 9 (25%) | 27 (75%) | 6 | 4.6 | 104 |

### Per question (correct runs / total)

| Question | Tests | Reference | off | warn |
|---|---|---|---|---|
| Q01: How many farm plots are in the dataset? | basic count | 24 | 3/3 | 3/3 |
| Q02: What is the EPSG code of the coordinate reference system of the plots layer? | CRS inspection | 4326 | 0/3 | 0/3 |
| Q03: How many farm plots have an invalid geometry? | geometry validity | 1 | 1/3 | 1/3 |
| Q04: What is the total area of rice plots in hectares? | filter + area; plots stored in degrees | 2.9815 | 3/3 | 3/3 |
| Q05: What is the total length of all roads in kilometres? | length; roads stored in degrees | 1.7923 | 3/3 | 2/3 |
| Q06: How many farm plots lie within 50 metres of the canal (touching the 50 m buffer counts)? | buffer in metres + overlay; canal and plots in different CRSs | 12 | 0/3 | 0/3 |
| Q07: Which village contains the most farm plots? | spatial join + group count; CRS mismatch | Bhawal | 0/3 | 0/3 |
| Q08: What is the total area in hectares of the farm plots in Kaliakoir village? | filter + join + area; includes the invalid plot | 1.8627 | 0/3 | 0/3 |
| Q09: How many soil samples fall inside wheat plots? | point-in-polygon | 4 | 0/3 | 0/3 |
| Q10: What is the mean area in hectares of the plots that received at least 120 kg N/ha? | numeric filter + mean area | 0.256 | 0/3 | 0/3 |
| Q11: What is the average of the mean NDVI values of the wheat plots? | zonal statistics + aggregation | 0.5901 | 0/3 | 0/3 |
| Q12: Which plot (plot_id) has the highest mean NDVI? | zonal statistics + argmax | P08 | 0/3 | 0/3 |

### Failure types (runs with a wrong or missing answer)

| Failure type | off | warn |
|---|---|---|
| answer_not_in_tool_output | 1 | 1 |
| crs_mismatch | 5 | 3 |
| llm_error | 2 | 1 |
| measured_in_degrees | 3 | 3 |
| no_answer_max_steps | 1 | 6 |
| wrong_workflow | 14 | 13 |

### What the results show

- **The small model is not reliable.** It answered 19 of 72 runs correctly. Only Q01, Q04
  and Q05 were right in most runs. Q06 to Q12 (spatial joins, a buffer, a numeric
  filter, zonal statistics) failed in every run.
- **The checks, shown to the model, did not help.** 9/36 correct with `warn` against 10/36
  with `off`. The model often read the warning, said what it should do, and then did
  something else. In all three `warn` runs of Q06 it received a `CRS_MISMATCH` error and an
  `EMPTY_RESULT` warning and still answered "0 plots".
- **The checks are useful as a diagnosis, not a correction.** Because every check is
  written to the trace in both modes, each wrong answer can be traced to a cause: a join
  across CRSs (8 runs), a measurement in degrees (6 runs), a skipped filter, and so on.
- **Most wrong answers were grounded in tool output.** In 39 of the 43 runs that finished
  with a wrong answer, the reported number appears in a tool result. The model mostly did
  not invent numbers; it ran the wrong workflow and reported its result faithfully.
  A check that only asks "did this number come from a tool?" would pass these answers.
- **Typical wrong workflows**, read from the traces:
  - Q02 (EPSG code): in all 6 runs the model inspected the layer, saw `EPSG:4326`, then
    reprojected and reported the total plot *area* instead. The system prompt's
    rule about areas probably pulled it towards measuring.
  - Q08 (Kaliakoir area): the runs that answered reported the area of *all* plots
    (5.68 ha, with the bow-tie plot unrepaired), skipping the village filter.
  - Q09 (samples in wheat plots): 13 in all 6 runs, which is the number of samples in
    *any* plot; the wheat filter was skipped.
  - Q12 (plot with highest NDVI): the runs that answered gave an NDVI value instead of
    the plot ID.

### The first run found a bug in the agent loop

The first full run (kept in `results/run1_no_fallback/` and `traces/eval_run1_no_fallback/`)
scored 7/36 in both modes. 20 of its runs ended with the model returning empty
replies until the step limit. Logging the raw Ollama responses showed why: the model was
writing tool calls Ollama could not parse (an invented tool `reproject_layer`, a made-up
argument `output_crs`), and Ollama dropped them silently. The agent loop then only said
"continue", so the model never learned that its call had failed.

The fix (`OllamaBackend._text_fallback`): when a reply is empty although the model
generated tokens, the agent asks again without the tool-call channel and reads the call
from plain text. A bad call now reaches the tools and gets an ordinary error message
("Unexpected argument(s) for reproject: ['layers', 'to_crs']. Allowed: ['epsg', 'layer',
'out_name']"). The fallback was used in 19 of the 72 runs above, and every trace records
it (`"text_fallback": true`). Runs ending at the step limit fell from 20 to 7.

This is the kind of silent failure the project is about: the agent's own plumbing, not the
model alone, decided how often the system answered at all. The first run's summary
uses an earlier version of the failure labels; see the commit history.

## Limitations

- **Synthetic data, small benchmark.** 12 questions on one made-up study area. The data are
  built to contain known traps, so the results say more about those traps than about GIS
  work in general.
- **One small model, few runs.** A 3B model on CPU makes mistakes a larger model may not.
  Three runs per question at temperature 0.2 is enough to see patterns, not to estimate
  rates precisely: 9/36 against 10/36 is no difference. Results depend on the model,
  the prompt and the temperature.
- **One agent design.** Running the model with a different loop (planning first, one tool
  per turn, retries) could change the results a lot. The first run shows how much the loop
  matters.
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
   the model's own arithmetic or memory. In the results above, 39 of 43 wrong answers
   passed this test, so grounding alone says little about correctness.

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
