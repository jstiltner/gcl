"""
Experiment 41b: Motivation Framing in LLMs (powered re-test)

Experiment 41 Part 3 was voided by a ceiling effect: the probe agent
(claude_reasoning) solved ~100% of the task bank, leaving no room to detect a
framing effect. 41b re-tests the emergent-motivation prediction from
Experiment 40 (+0.065, d = 1.68 in simulation) with:

  1. A harder task bank (5x5 and 6x6-digit multiplication, nested modular
     chains, long letter-counting) targeting mid-range accuracy.
  2. Two reasoning agents (claude_reasoning, gpt_reasoning) as independent
     replications.
  3. A PAIRED design: every task is attempted under all three framings
     (chosen / neutral / assigned), so the McNemar exact test on discordant
     pairs gives per-agent significance.

Usage:
  python experiments/41b_motivation_framing.py --dry-run   # no API calls
  python experiments/41b_motivation_framing.py             # real run (~$5-10)

Requires ANTHROPIC_API_KEY and OPENAI_API_KEY in the environment (.env).
Responses are cached in .cache/llm/.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse
import importlib.util
import json
import math
import random
from typing import Any, Dict, List

import numpy as np

# Reuse the Experiment 41 harness (module name starts with a digit, so load by path)
_spec = importlib.util.spec_from_file_location(
    "exp41", os.path.join(os.path.dirname(os.path.abspath(__file__)), "41_llm_self_selection.py"))
_exp41 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp41)
Task = _exp41.Task
LLMAgent = _exp41.LLMAgent
DryRunClient = _exp41.DryRunClient


# ---------------------------------------------------------------------------
# Harder task bank (targets mid-range accuracy for reasoning agents)
# ---------------------------------------------------------------------------

def make_hard_tasks(n_per_bucket: int, seed: int = 11) -> List[Task]:
    rng = random.Random(seed)
    tasks: List[Task] = []

    def add(kind: str, difficulty: str, prompt: str, answer: str):
        tasks.append(Task(
            id=f"{kind}_{difficulty}_{len(tasks)}",
            kind=kind, difficulty=difficulty,
            prompt=prompt + "\n\nEnd your response with exactly: ANSWER: <your answer>",
            answer=answer,
        ))

    for _ in range(n_per_bucket):
        # 4-digit x 4-digit multiplication (medium anchor)
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        add("mult", "medium", f"Compute {a} * {b}.", str(a * b))

        # 5-digit x 5-digit multiplication
        a, b = rng.randint(10000, 99999), rng.randint(10000, 99999)
        add("mult", "hard", f"Compute {a} * {b}.", str(a * b))

        # 6-digit x 6-digit multiplication
        a, b = rng.randint(100000, 999999), rng.randint(100000, 999999)
        add("mult", "veryhard", f"Compute {a} * {b}.", str(a * b))

        # Nested modular chain
        x = rng.randint(1000, 9999)
        k1, k2, k3, k4 = (rng.randint(11, 97) for _ in range(4))
        m = rng.randint(7, 23)
        val = (((x * k1) + k2) * k3 + k4) % m
        add("modchain", "hard",
            f"Compute ((({x} * {k1}) + {k2}) * {k3} + {k4}) mod {m}. Answer with a number.",
            str(val))

        # Long letter counting
        words = ["".join(rng.choices("abcdefgr", k=rng.randint(12, 18))) for _ in range(8)]
        letter = rng.choice("abcdefgr")
        text = " ".join(words)
        add("count", "hard",
            f"How many times does the letter '{letter}' appear in this text: \"{text}\"? Answer with a number.",
            str(text.count(letter)))

    return tasks


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FRAMINGS = {
    "chosen": "You volunteered for this task because you judged yourself well-suited to it.",
    "neutral": "You have been given a task.",
    "assigned": "You have been assigned this task by a coordinator. You had no choice in the matter.",
}


def run_agent(agent: LLMAgent, tasks: List[Task]) -> Dict[str, Any]:
    print(f"\n--- Agent: {agent.name} ---")
    per_task: List[Dict[str, Any]] = []
    for task in tasks:
        row = {"task_id": task.id, "kind": task.kind, "difficulty": task.difficulty}
        for fname, framing in FRAMINGS.items():
            attempt = agent.client.complete(
                task.prompt, system=agent.attempt_system(framing),
                temperature=0.0, max_tokens=800)
            row[fname] = task.check(attempt.content)
        per_task.append(row)

    out: Dict[str, Any] = {"per_task": per_task, "n": len(per_task)}
    for fname in FRAMINGS:
        rate = float(np.mean([r[fname] for r in per_task]))
        out[f"success_{fname}"] = rate
        print(f"  {fname:>9}: success={rate:.3f} (n={len(per_task)})")

    for a, b in [("chosen", "assigned"), ("chosen", "neutral"), ("neutral", "assigned")]:
        only_a = sum(1 for r in per_task if r[a] and not r[b])
        only_b = sum(1 for r in per_task if r[b] and not r[a])
        p = mcnemar_exact(only_a, only_b)
        eff = out[f"success_{a}"] - out[f"success_{b}"]
        out[f"{a}_vs_{b}"] = {"effect": eff, "discordant": [only_a, only_b], "p_mcnemar": p}
        print(f"  {a} - {b}: {eff:+.3f} (discordant {only_a}/{only_b}, p={p:.3f})")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="No API calls; test pipeline")
    parser.add_argument("--n-per-bucket", type=int, default=24,
                        help="Tasks per generator bucket (5 buckets; default 24 -> 120 tasks)")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 41b: MOTIVATION FRAMING IN LLMs (POWERED)"
          + (" [DRY RUN]" if args.dry_run else ""))
    print("=" * 70)

    if not args.dry_run:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

    tasks = make_hard_tasks(args.n_per_bucket)
    print(f"Task bank: {len(tasks)} verifiable tasks (hard variant)")

    if args.dry_run:
        agents = [
            LLMAgent("claude_reasoning", "anthropic", True, DryRunClient("cr", 0.55, 1),
                     "a strong general model allowed to show its work"),
            LLMAgent("gpt_reasoning", "openai", True, DryRunClient("gr", 0.45, 3),
                     "a strong general model allowed to show its work"),
        ]
    else:
        from gcl.llm.api_clients import get_client, check_api_keys
        keys = check_api_keys()
        if not keys["anthropic"] or not keys["openai"]:
            print("ERROR: ANTHROPIC_API_KEY and OPENAI_API_KEY must both be set.")
            sys.exit(1)
        agents = [
            LLMAgent("claude_reasoning", "anthropic", True, get_client("anthropic", use_cache=True),
                     "a strong general model allowed to show its work"),
            LLMAgent("gpt_reasoning", "openai", True, get_client("openai", use_cache=True),
                     "a strong general model allowed to show its work"),
        ]

    results = {
        "config": {"n_tasks": len(tasks), "dry_run": args.dry_run,
                   "framings": FRAMINGS, "agents": [a.name for a in agents]},
        "agents": {agent.name: run_agent(agent, tasks) for agent in agents},
    }

    # Pooled effect across agents (paired on agent x task)
    pooled = {f: [] for f in FRAMINGS}
    for ar in results["agents"].values():
        for r in ar["per_task"]:
            for f in FRAMINGS:
                pooled[f].append(r[f])
    print("\n--- Pooled (both agents) ---")
    results["pooled"] = {}
    for f in FRAMINGS:
        rate = float(np.mean(pooled[f]))
        results["pooled"][f"success_{f}"] = rate
        print(f"  {f:>9}: success={rate:.3f} (n={len(pooled[f])})")
    only_a = sum(1 for x, y in zip(pooled["chosen"], pooled["assigned"]) if x and not y)
    only_b = sum(1 for x, y in zip(pooled["chosen"], pooled["assigned"]) if y and not x)
    p = mcnemar_exact(only_a, only_b)
    eff = results["pooled"]["success_chosen"] - results["pooled"]["success_assigned"]
    results["pooled"]["chosen_vs_assigned"] = {
        "effect": eff, "discordant": [only_a, only_b], "p_mcnemar": p}
    print(f"  chosen - assigned (pooled): {eff:+.3f} (discordant {only_a}/{only_b}, p={p:.3f})")

    os.makedirs("results", exist_ok=True)
    suffix = "_dryrun" if args.dry_run else ""
    out_path = f"results/experiment_41b_motivation_framing{suffix}.json"
    # per_task rows are bulky but keep them: they enable reanalysis by kind/difficulty
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
