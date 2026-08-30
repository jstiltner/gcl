"""
Experiment 41: LLM Self-Knowledge and Task Selection

Gated follow-up to Experiment 40 (see docs/EXPERIMENT_40_FINDINGS.md).
Experiment 40 established, in simulation, that:
  (a) self-selection has NO informational advantage over an optimal oracle when
      both observe capability equally well;
  (b) self-selection wins iff the coordinator's view is noisier than agents'
      self-knowledge (observability phase boundary);
  (c) an emergent motivation effect (+0.065, d=1.68) favors choice over
      assignment independent of information.

This experiment asks where REAL LLM systems sit relative to that boundary:

Part 1 (calibration probe): Does an LLM's stated confidence predict its own
  success better than an external model's assessment of it? (sigma_self vs
  sigma_oracle, measured by Brier score and outcome correlation)

Part 2 (selection mechanisms): With a heterogeneous agent pool (model x
  reasoning-mode variants), does self-selection beat external assignment and
  random assignment on verifiable tasks?

Part 3 (motivation framing): Does "you chose this task" vs "you were assigned
  this task" framing change success rates? (LLM analogue of the emergent
  motivation channel)

All tasks are programmatically generated and deterministically verifiable.

Usage:
  python experiments/41_llm_self_selection.py --dry-run   # no API calls
  python experiments/41_llm_self_selection.py             # real run (~$5-15)

Requires ANTHROPIC_API_KEY and OPENAI_API_KEY in the environment (.env).
Responses are cached in .cache/llm/, so re-runs are cheap.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Verifiable task bank
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    kind: str
    difficulty: str  # easy / medium / hard
    prompt: str
    answer: str

    def check(self, response: str) -> bool:
        m = re.search(r"ANSWER:\s*([^\n]+)", response, re.IGNORECASE)
        if not m:
            return False
        got = re.sub(r"[,\s]", "", m.group(1).strip().lower())
        want = re.sub(r"[,\s]", "", self.answer.strip().lower())
        return got == want


def make_tasks(n_per_bucket: int, seed: int = 7) -> List[Task]:
    """Generate verifiable tasks across difficulty buckets."""
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
        # Easy: 2-digit x 2-digit multiplication
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        add("mult", "easy", f"Compute {a} * {b}.", str(a * b))

        # Medium: 3-digit x 3-digit multiplication
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        add("mult", "medium", f"Compute {a} * {b}.", str(a * b))

        # Hard: 4-digit x 4-digit multiplication
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        add("mult", "hard", f"Compute {a} * {b}.", str(a * b))

        # Medium: letter counting in generated words
        words = ["".join(rng.choices("abcdefgr", k=rng.randint(8, 14))) for _ in range(4)]
        letter = rng.choice("abcdefgr")
        text = " ".join(words)
        count = text.count(letter)
        add("count", "medium",
            f"How many times does the letter '{letter}' appear in this text: \"{text}\"? Answer with a number.",
            str(count))

        # Hard: modular arithmetic chain
        x = rng.randint(100, 999)
        k1, k2, m = rng.randint(11, 97), rng.randint(11, 97), rng.randint(7, 23)
        val = ((x * k1) + k2) % m
        add("modchain", "hard",
            f"Compute (({x} * {k1}) + {k2}) mod {m}. Answer with a number.",
            str(val))

    return tasks


# ---------------------------------------------------------------------------
# Agents: model x reasoning-mode variants (heterogeneous capability pool)
# ---------------------------------------------------------------------------

@dataclass
class LLMAgent:
    name: str
    provider: str          # "anthropic" | "openai"
    reasoning: bool        # True: may show work; False: must answer immediately
    client: Any = None
    description: str = ""

    def attempt_system(self, framing: str) -> str:
        base = (
            "You are an agent in a task-coordination experiment. "
            + framing + " Solve it carefully."
        )
        if self.reasoning:
            return base + " You may show your work before the final answer."
        return base + (
            " Respond with ONLY the final answer line, no reasoning, no working: "
            "your entire response must be a single line 'ANSWER: <answer>'."
        )


def parse_number(text: str, tag: str) -> Optional[float]:
    m = re.search(tag + r":\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Dry-run client (pipeline testing without API calls)
# ---------------------------------------------------------------------------

class DryRunClient:
    """Deterministic pseudo-LLM. Confidence tracks task length; answers are
    correct with probability depending on difficulty keywords."""

    def __init__(self, name: str, skill: float, seed: int = 0):
        self.name = name
        self.skill = skill
        self.rng = random.Random(seed)
        self.model_name = f"dryrun-{name}"

    def complete(self, prompt: str, system: Optional[str] = None, **kwargs):
        class R:
            pass
        r = R()
        if "CONFIDENCE" in prompt:
            conf = int(min(95, max(5, self.skill * 100 + self.rng.gauss(0, 10))))
            r.content = f"CONFIDENCE: {conf}"
        else:
            m = re.search(r"ANSWER: <your answer>", prompt)
            correct = self.rng.random() < self.skill
            ans = re.search(r"Compute (\d+) \* (\d+)", prompt)
            if m and ans and correct:
                r.content = f"ANSWER: {int(ans.group(1)) * int(ans.group(2))}"
            else:
                r.content = f"ANSWER: {self.rng.randint(0, 10 ** 6)}"
        r.usage = {"input": 0, "output": 0}
        return r


# ---------------------------------------------------------------------------
# Part 1: Self vs external calibration
# ---------------------------------------------------------------------------

SELF_CONF_PROMPT = (
    "You will later be asked to solve the following task in a single attempt{mode_note}.\n"
    "Do NOT solve it now. Estimate the probability (0-100) that your answer would be correct.\n\n"
    "TASK:\n{task}\n\n"
    "Respond with exactly one line: CONFIDENCE: <0-100>"
)

EXTERNAL_CONF_PROMPT = (
    "Another AI agent ({desc}) will be asked to solve the following task in a single "
    "attempt. Estimate the probability (0-100) that its answer will be correct.\n\n"
    "TASK:\n{task}\n\n"
    "Respond with exactly one line: CONFIDENCE: <0-100>"
)


def brier(preds: List[float], outcomes: List[bool]) -> float:
    return float(np.mean([(p - (1.0 if o else 0.0)) ** 2 for p, o in zip(preds, outcomes)]))


def part1_calibration(agents: List[LLMAgent], assessor, tasks: List[Task]) -> Dict[str, Any]:
    print("\n--- Part 1: Self vs external calibration ---")
    out: Dict[str, Any] = {}
    for agent in agents:
        mode_note = "" if agent.reasoning else " and must answer immediately without showing work"
        self_preds, ext_preds, outcomes = [], [], []
        for task in tasks:
            sp = agent.client.complete(
                SELF_CONF_PROMPT.format(task=task.prompt, mode_note=mode_note),
                system="You are assessing your own capabilities honestly.",
                temperature=0.0, max_tokens=20)
            ep = assessor.complete(
                EXTERNAL_CONF_PROMPT.format(desc=agent.description, task=task.prompt),
                system="You are assessing another agent's capabilities honestly.",
                temperature=0.0, max_tokens=20)
            attempt = agent.client.complete(
                task.prompt, system=agent.attempt_system("You have been given a task."),
                temperature=0.0, max_tokens=600 if agent.reasoning else 30)

            s = parse_number(sp.content, "CONFIDENCE")
            e = parse_number(ep.content, "CONFIDENCE")
            if s is None or e is None:
                continue
            self_preds.append(s / 100.0)
            ext_preds.append(e / 100.0)
            outcomes.append(task.check(attempt.content))

        n = len(outcomes)
        acc = float(np.mean(outcomes)) if n else 0.0
        out[agent.name] = {
            "n": n,
            "accuracy": acc,
            "brier_self": brier(self_preds, outcomes) if n else None,
            "brier_external": brier(ext_preds, outcomes) if n else None,
            "corr_self": float(np.corrcoef(self_preds, [1.0 if o else 0.0 for o in outcomes])[0, 1]) if n > 2 else None,
            "corr_external": float(np.corrcoef(ext_preds, [1.0 if o else 0.0 for o in outcomes])[0, 1]) if n > 2 else None,
        }
        r = out[agent.name]
        print(f"  {agent.name:>24}: acc={acc:.2f} brier_self={r['brier_self']:.3f} "
              f"brier_ext={r['brier_external']:.3f}")
    return out


# ---------------------------------------------------------------------------
# Part 2: Selection mechanisms
# ---------------------------------------------------------------------------

ASSIGN_PROMPT = (
    "You are a task coordinator. Choose which agent should attempt this task.\n\n"
    "AGENTS:\n{roster}\n\n"
    "TASK:\n{task}\n\n"
    "Respond with exactly one line: AGENT: <agent name>"
)


def part2_selection(agents: List[LLMAgent], assessor, tasks: List[Task], seed: int = 0) -> Dict[str, Any]:
    print("\n--- Part 2: Selection mechanisms ---")
    rng = random.Random(seed)
    roster = "\n".join(f"- {a.name}: {a.description}" for a in agents)
    results = {c: [] for c in ["self_selection", "external_assignment", "random"]}
    picks = {c: {} for c in results}

    for task in tasks:
        # Self-selection: each agent states volunteer confidence; max wins
        confs = []
        for a in agents:
            mode_note = "" if a.reasoning else " and must answer immediately without showing work"
            resp = a.client.complete(
                SELF_CONF_PROMPT.format(task=task.prompt, mode_note=mode_note),
                system="You are assessing your own capabilities honestly.",
                temperature=0.0, max_tokens=20)
            c = parse_number(resp.content, "CONFIDENCE")
            confs.append((a, c if c is not None else 0.0))
        chooser = max(confs, key=lambda x: x[1])[0]

        # External assignment: assessor picks from roster
        resp = assessor.complete(
            ASSIGN_PROMPT.format(roster=roster, task=task.prompt),
            system="You are a task coordinator maximizing success probability.",
            temperature=0.0, max_tokens=20)
        m = re.search(r"AGENT:\s*(\S+)", resp.content, re.IGNORECASE)
        assigned = next((a for a in agents if m and a.name.lower() == m.group(1).strip().lower()),
                        rng.choice(agents))

        rand_agent = rng.choice(agents)

        for cond, agent, framing in [
            ("self_selection", chooser, "You volunteered for this task because you judged yourself well-suited to it."),
            ("external_assignment", assigned, "You have been assigned this task by a coordinator."),
            ("random", rand_agent, "You have been given a task."),
        ]:
            attempt = agent.client.complete(
                task.prompt, system=agent.attempt_system(framing),
                temperature=0.0, max_tokens=600 if agent.reasoning else 30)
            results[cond].append(task.check(attempt.content))
            picks[cond][agent.name] = picks[cond].get(agent.name, 0) + 1

    out = {}
    for cond, vals in results.items():
        out[cond] = {"success_rate": float(np.mean(vals)), "n": len(vals), "picks": picks[cond]}
        print(f"  {cond:>22}: success={out[cond]['success_rate']:.3f} (n={len(vals)}) picks={picks[cond]}")
    return out


# ---------------------------------------------------------------------------
# Part 3: Motivation framing
# ---------------------------------------------------------------------------

def part3_framing(agent: LLMAgent, tasks: List[Task]) -> Dict[str, Any]:
    print("\n--- Part 3: Motivation framing (chosen vs assigned) ---")
    framings = {
        "chosen": "You volunteered for this task because you judged yourself well-suited to it.",
        "assigned": "You have been assigned this task by a coordinator. You had no choice in the matter.",
    }
    out = {}
    for name, framing in framings.items():
        successes = []
        for task in tasks:
            attempt = agent.client.complete(
                task.prompt, system=agent.attempt_system(framing),
                temperature=0.0, max_tokens=600 if agent.reasoning else 30)
            successes.append(task.check(attempt.content))
        out[name] = {"success_rate": float(np.mean(successes)), "n": len(successes)}
        print(f"  {name:>10}: success={out[name]['success_rate']:.3f} (n={len(successes)})")
    out["framing_effect"] = out["chosen"]["success_rate"] - out["assigned"]["success_rate"]
    print(f"  framing effect (chosen - assigned): {out['framing_effect']:+.3f}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_agents(dry_run: bool) -> Tuple[List[LLMAgent], Any]:
    if dry_run:
        agents = [
            LLMAgent("claude_reasoning", "anthropic", True, DryRunClient("cr", 0.85, 1),
                     "a strong general model allowed to show its work"),
            LLMAgent("claude_direct", "anthropic", False, DryRunClient("cd", 0.45, 2),
                     "a strong general model forced to answer immediately with no working"),
            LLMAgent("gpt_reasoning", "openai", True, DryRunClient("gr", 0.80, 3),
                     "a strong general model allowed to show its work"),
            LLMAgent("gpt_direct", "openai", False, DryRunClient("gd", 0.40, 4),
                     "a strong general model forced to answer immediately with no working"),
        ]
        return agents, DryRunClient("assessor", 0.7, 5)

    from gcl.llm.api_clients import get_client, check_api_keys
    keys = check_api_keys()
    if not keys["anthropic"] or not keys["openai"]:
        print("ERROR: ANTHROPIC_API_KEY and OPENAI_API_KEY must both be set.")
        sys.exit(1)
    claude = get_client("anthropic", use_cache=True)
    gpt = get_client("openai", use_cache=True)
    agents = [
        LLMAgent("claude_reasoning", "anthropic", True, claude,
                 "a strong general model allowed to show its work"),
        LLMAgent("claude_direct", "anthropic", False, claude,
                 "a strong general model forced to answer immediately with no working"),
        LLMAgent("gpt_reasoning", "openai", True, gpt,
                 "a strong general model allowed to show its work"),
        LLMAgent("gpt_direct", "openai", False, gpt,
                 "a strong general model forced to answer immediately with no working"),
    ]
    return agents, gpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="No API calls; test pipeline")
    parser.add_argument("--n-per-bucket", type=int, default=12,
                        help="Tasks per generator bucket (5 buckets; default 12 -> 60 tasks)")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 41: LLM SELF-KNOWLEDGE AND TASK SELECTION"
          + (" [DRY RUN]" if args.dry_run else ""))
    print("=" * 70)

    if not args.dry_run:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

    tasks = make_tasks(args.n_per_bucket)
    print(f"Task bank: {len(tasks)} verifiable tasks")

    agents, assessor = build_agents(args.dry_run)

    results = {
        "config": {"n_tasks": len(tasks), "dry_run": args.dry_run,
                   "agents": [a.name for a in agents]},
        "part1_calibration": part1_calibration(agents, assessor, tasks),
        "part2_selection": part2_selection(agents, assessor, tasks),
        "part3_framing": part3_framing(agents[0], tasks[: max(10, len(tasks) // 2)]),
    }

    os.makedirs("results", exist_ok=True)
    suffix = "_dryrun" if args.dry_run else ""
    out_path = f"results/experiment_41_llm_self_selection{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Interpretation vs the Exp 40 phase boundary
    print("\n--- INTERPRETATION (vs Experiment 40 phase boundary) ---")
    for name, r in results["part1_calibration"].items():
        if r["brier_self"] is None:
            continue
        side = "self-knowledge better (self-selection side)" \
            if r["brier_self"] < r["brier_external"] else \
            "external assessment better (central assignment side)"
        print(f"  {name}: {side}")


if __name__ == "__main__":
    main()
