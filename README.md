# Grounded Commitment Learning (GCL)

**A formal framework for AI coordination under semantic drift.**

[![Tests](https://img.shields.io/badge/tests-382%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**CI-reproduced, on every push to `main`** — [![Punishment Paradox](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjstiltner%2Fgcl%2Fmain%2Fci_results%2Fbadge-punishment-paradox.json)](.github/workflows/ci-reproduce.yml) [![Hart-Moore](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjstiltner%2Fgcl%2Fmain%2Fci_results%2Fbadge-hart-moore.json)](.github/workflows/ci-reproduce.yml)
— reduced-scale (n=5 seeds) run of the real simulation, not a copy of the published numbers;
see [`ci_results/latest.json`](ci_results/latest.json) and
[`experiments/derive_real_headline_stats.py`](experiments/derive_real_headline_stats.py).
Full-scale (n=30) reproduction: [open the Colab notebook](notebooks/reproduce_headline_stats.ipynb).

> **All quantitative claims in this repository are indexed in
> [`docs/CLAIMS.md`](docs/CLAIMS.md)**, including effect sizes, confidence
> intervals, and two retractions from our own revision process.

## The Core Insight

> **GCL completes existing AI coordination approaches, it doesn't replace them.**

Natural language coordination works well when action semantics are stable. But when models update, APIs change, or environments shift, the meaning of "help the user" or "be accurate" can drift. GCL provides a **verification layer** that detects and adapts to this drift.

**The key finding**: GCL's advantage under drift becomes statistically significant at drift rates ε ≥ 0.15 (+0.027 to +0.036 success rate). At zero drift GCL slightly *underperforms* a natural-language baseline (−0.060), and at ε ∈ {0.05, 0.10} differences are not significant — grounded verification is a hedge against drift, not a free lunch (Experiment 09; see [`docs/CLAIMS.md`](docs/CLAIMS.md) #11).

## What is GCL?

GCL is a **minimal formal calculus** for AI coordination based on verifiable commitments:

```
C = (τ, a, φ, F, σ)

τ: Trigger predicate (when does this activate?)
a: Action function (what is promised?)
φ: Verification predicate (was it fulfilled?)
F: Failure modes (how can it fail?)
σ: Stake (what is risked?)
```

Three operations: **ISSUE**, **VERIFY**, **SETTLE**

Three compositions: **Sequential (;)**, **Parallel (‖)**, **Conditional (◁p▷)**

This is teachable in 2 pages and forms a reusable abstraction. See [Appendix A](docs/APPENDIX_A_FORMAL_CALCULUS.md) for the complete formal specification.

## When to Use GCL

GCL is designed for **multi-agent coordination under semantic drift**. Use it when:

| Requirement | Description |
|-------------|-------------|
| ✓ Verifiable outcomes | Success/failure can be objectively determined |
| ✓ Repeated interactions | Agents learn from past commitment outcomes |
| ✓ Meaningful stake | Agents have resources they care about |
| ✓ Semantic drift | Action meanings change over time |

**Don't use GCL when**: outcomes are subjective, interactions are one-shot, or semantics are perfectly stable. See [`src/gcl/core/scope.py`](src/gcl/core/scope.py) for detailed applicability assessment.

## Quick Start

```python
from gcl.core.calculus import Commitment, FailureMode, issue, verify, settle

# Define a commitment
commitment = Commitment(
    trigger=lambda s: s.get("needs_review", False),
    action=lambda s: "review_code",
    verification=lambda s, s_, a: s_.get("review_complete", False),
    failures=[
        FailureMode(lambda s, s_, a: s_.get("timeout", False), 0.8, "timeout"),
        FailureMode(lambda s, s_, a: not s_.get("review_complete", False), 0.5, "incomplete"),
    ],
    stake=1.0,
)

# Issue the commitment
issued = issue(commitment, issuer="code_reviewer")

# After action execution, verify outcome
pre_state = {"needs_review": True}
post_state = {"review_complete": True}
outcome = verify(issued, pre_state, post_state, "review_code")

# Settle stake based on outcome
reward = settle(issued, outcome)
print(f"Outcome: {outcome.status.value}, Reward: {reward}")
```

## Architecture

GCL has a 4-level architecture:

```
Level 4: Multi-Agent Coordination
         ├── Shared commitment verification
         └── Emergent coordination protocols

Level 3: Commitment-Grounded RL
         ├── Policy gradient over commitment selection
         └── Confidence-weighted exploration

Level 2: Template Hierarchy
         ├── Parameterized commitment generators
         └── Domain-specific templates

Level 1: Grounded Commitments (Minimal Calculus)
         ├── 5-tuple: (τ, a, φ, F, σ)
         ├── 3 operations: ISSUE, VERIFY, SETTLE
         └── 3 compositions: ;, ‖, ◁p▷
```

## Key Results

### Theoretical Foundations
- **6 empirically validated propositions** (validated via simulation and property tests, not formal mathematical proofs)
- Closure under composition
- Convergence behavior for policy learning
- Nash-equilibrium-consistent behavior in multi-agent settings

### Empirical Findings
- **Emergent motivation**: self-selection with emergent effort beats a truly optimal assignment oracle by +0.065 (d = 1.68); with effort held fixed the advantage is exactly zero — choice creates commitment, not information (Experiment 40)
- **Observability phase boundary**: self-selection wins iff the coordinator's view of agent capability is noisier than agents' self-knowledge (Experiment 40, Part B)
- **Real LLMs sit on the central-assignment side of that boundary**: for all 4 LLM agents tested, an external assessor predicted their success better than their own stated confidence; confidence-based self-selection rewards overconfidence, implying selection should be grounded in verified track records (as GCL's reputation mechanism does), not self-reports (Experiment 41)
- **Drift threshold**: GCL advantage significant for ε ≥ 0.15 (Experiment 09)
- **vs. MARL baselines**: GCL is **third of five** on final cooperation (IQL 0.553 > QMIX 0.542 > GCL 0.534 > MAPPO 0.522 > random 0.475) and needs no training to get there (Experiment 36). *An earlier version of this line read "~97% of MARL performance with 25–50× fewer episodes"; both figures were retracted on 2026-09-09 — see [`docs/CLAIMS.md`](docs/CLAIMS.md) rows 9 and 10.*
- **Population dynamics**: Protocol convergence at 50-500 agents; note Experiment 07 passes **3 of 4** predictions — Template Replicator Dynamics fails all three sub-checks (Experiment 07)

### Practical Demonstrations
- CI/CD pipeline coordination
- Multi-LLM code review
- Heterogeneous agent coordination

## Project Structure

```
gcl/
├── src/gcl/
│   ├── core/
│   │   ├── calculus.py      # Minimal commitment calculus
│   │   ├── commitment.py    # Core dataclasses
│   │   ├── scope.py         # Applicability assessment
│   │   └── predicates.py    # Safe expression parser
│   ├── learning/            # RL components
│   ├── multiagent/          # Multi-agent coordination
│   ├── population/          # Population-scale dynamics
│   └── baselines/           # Comparison baselines
├── experiments/
│   ├── 09_drift_threshold.py    # Drift crossover experiment
│   └── 10_cicd_coordination.py  # CI/CD demo
├── tests/                   # 382 tests
├── docs/
│   └── APPENDIX_A_FORMAL_CALCULUS.md  # Formal specification
└── results/                 # Experiment outputs
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gcl.git
cd gcl

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Running Experiments

```bash
# Drift threshold experiment
python experiments/09_drift_threshold.py

# CI/CD coordination demo
python experiments/10_cicd_coordination.py

# Population dynamics
python experiments/07_population_dynamics.py
```

## Known Limitations

GCL has explicit boundaries. See [`src/gcl/core/scope.py`](src/gcl/core/scope.py) for details:

1. **Cold Start Problem**: New commitments have no history
2. **Verification Oracle**: Assumes verification is correct and computable
3. **Adversarial Robustness**: Can be gamed by strategic agents
4. **Scalability**: O(n² × m) for n agents and m commitment types
5. **Non-Stationarity**: Historical confidence may lag environmental changes
6. **Partial Observability**: Requires observable state for verification

## Research Directions

- Meta-learning for commitment confidence
- Learned verification from examples
- Game-theoretic adversarial analysis
- Scalable approximate confidence tracking
- Adaptive forgetting for non-stationarity

## Research Integrity & Revision History

This project treats its own claims adversarially. Two headline findings were
retracted after internal red-teaming, and the corrections are documented rather
than erased:

1. **Exp 34A → 39**: an "effort > information" mechanism decomposition was
   retracted after we realized the effort effect was a hardcoded parameter,
   not emergent behavior.
2. **Exp 39 → 40**: the "self-selection beats optimal matching by 81% via
   information asymmetry" claim was retracted after we found the comparison
   oracle optimized an anti-overkill objective that was strictly suboptimal
   under the success model. Against a truly optimal oracle, the information
   advantage is +0.000 — but a real emergent-motivation effect (+0.065,
   d = 1.68) survives, and an observability phase boundary determines when
   decentralized self-selection beats centralized assignment.
3. **Exp 40 → 41**: the corrected theory makes a falsifiable prediction —
   self-selection only wins where self-knowledge beats external observation.
   Testing it on real LLMs (Claude, GPT) showed they sit on the
   central-assignment side of the boundary: external assessment was better
   calibrated than self-confidence for every agent tested, and a powered
   re-test (Exp 41b) found no significant "chosen vs assigned" framing effect
   — the simulated motivation mechanism does not measurably transfer to
   prompted LLMs.

The full audit trail lives in [`docs/CLAIMS.md`](docs/CLAIMS.md),
[`docs/EXPERIMENT_40_FINDINGS.md`](docs/EXPERIMENT_40_FINDINGS.md), and
[`docs/EXPERIMENT_41_FINDINGS.md`](docs/EXPERIMENT_41_FINDINGS.md).

## Citation

```bibtex
@software{gcl2026,
  title = {Grounded Commitment Learning: A Formal Framework for AI Coordination},
  author = {Stiltner, Jason},
  year = {2026},
  url = {https://github.com/jstiltner/gcl}
}
```

## License

MIT

## Author

Jason Stiltner

---

*GCL: Because coordination under drift requires more than good intentions—it requires verification.*
