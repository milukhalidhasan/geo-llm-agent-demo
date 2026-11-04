"""Ask the agent one question from the command line.

    python ask.py "What is the total area of rice plots in hectares?"
    python ask.py "How many plots are within 50 m of the canal?" --checks off
    python ask.py "..." --model llama3.2:3b
"""
from __future__ import annotations

import argparse

from geo_agent.agent import Agent
from geo_agent.llm import OllamaBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--checks", default="warn", choices=["off", "warn", "block"])
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    agent = Agent(OllamaBackend(model=args.model, temperature=args.temperature),
                  check_mode=args.checks, max_steps=args.max_steps, verbose=True)
    res = agent.run(args.question)
    print()
    print(f"Answer : {res.answer_text}")
    print(f"Stopped: {res.stop_reason} after {res.n_tool_calls} tool calls, {res.elapsed_s:.0f} s")
    print(f"Trace  : {res.trace_path}")


if __name__ == "__main__":
    main()
