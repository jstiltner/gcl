# Appendix A: Minimal Commitment Calculus

## A.1 Formal Definitions

### Definition 1 (Commitment)
A **commitment** is a 5-tuple:

$$C = (\tau, a, \phi, F, \sigma)$$

where:
- $\tau: S \to \{0, 1\}$ is the **trigger predicate** (when commitment activates)
- $a: S \to A$ is the **action function** (what the agent commits to do)
- $\phi: S \times S \times A \to \{0, 1\}$ is the **verification predicate** (success condition)
- $F = \{(f_i, s_i, d_i)\}_{i=1}^n$ is the **failure mode set** (ordered by severity $s_i \in [0,1]$)
- $\sigma \in \mathbb{R}^+$ is the **stake** (resources at risk)

### Definition 2 (Failure Mode)
A **failure mode** is a triple $(f, s, d)$ where:
- $f: S \times S \times A \to \{0, 1\}$ is a predicate identifying this failure type
- $s \in [0, 1]$ is the severity (1.0 = catastrophic, 0.0 = minor)
- $d$ is a human-readable description

### Definition 3 (Outcome)
The **outcome function** $O: C \times S \times S \times A \to \{\text{SUCCESS}, \text{FAILURE}_i\}$ is:

$$O(C, s, s', a) = \begin{cases}
\text{SUCCESS} & \text{if } \phi(s, s', a) = 1 \\
\text{FAILURE}_i & \text{if } \phi(s, s', a) = 0 \land f_i(s, s', a) = 1 \land \forall_{j<i} f_j(s, s', a) = 0
\end{cases}$$

Failure modes are checked in severity order; the first matching mode determines the outcome.

---

## A.2 Operations

### Operation 1: ISSUE
$$\text{ISSUE}(C, \text{agent}) \to C'$$

Registers commitment $C$ as active, assigning a unique identifier and recording the issuing agent. This operation is idempotent—issuing the same commitment twice has no additional effect.

### Operation 2: VERIFY
$$\text{VERIFY}(C, s, s', a) \to O$$

Evaluates the commitment against observed state transition $(s, s')$ and action $a$, returning an outcome $O \in \{\text{SUCCESS}, \text{FAILURE}_i\}$.

**Verification is deterministic**: Given the same inputs, VERIFY always returns the same outcome.

### Operation 3: SETTLE
$$\text{SETTLE}(C, O) \to \Delta\sigma$$

Computes stake adjustment based on outcome:

$$\Delta\sigma = \begin{cases}
0 & \text{if } O = \text{SUCCESS} \\
-\sigma \cdot s_i & \text{if } O = \text{FAILURE}_i
\end{cases}$$

The stake reduction is proportional to failure severity.

---

## A.3 Composition Operators

### Sequential Composition (;)
$$C_1 ; C_2 = (\tau_1, a_{seq}, \phi_{seq}, F_1 \cup F_2, \sigma_1 + \sigma_2)$$

where:
- $a_{seq}(s) = a_2(s')$ after $a_1(s)$ produces $s'$
- $\phi_{seq}(s, s'', a) = \phi_1(s, s', a_1) \land \phi_2(s', s'', a_2)$

**Semantics**: $C_2$ activates only after $C_1$ succeeds.

### Parallel Composition (‖)
$$C_1 \| C_2 = (\tau_1 \land \tau_2, a_{par}, \phi_{par}, F_1 \cup F_2, \max(\sigma_1, \sigma_2))$$

where:
- $a_{par}(s) = (a_1(s), a_2(s))$ (both actions execute)
- $\phi_{par}(s, s', a) = \phi_1(s, s', a_1) \land \phi_2(s, s', a_2)$

**Semantics**: Both commitments must succeed for the composition to succeed.

### Conditional Composition (◁p▷)
$$C_1 \triangleleft p \triangleright C_2 = (\tau_1 \lor \tau_2, a_{cond}, \phi_{cond}, F_1 \cup F_2, \max(\sigma_1, \sigma_2))$$

where:
- $a_{cond}(s) = \begin{cases} a_1(s) & \text{if } p(s) = 1 \\ a_2(s) & \text{otherwise} \end{cases}$
- $\phi_{cond}$ follows the same branching logic

**Semantics**: Predicate $p$ determines which branch executes.

---

## A.4 Theorems

### Theorem 1 (Closure)
The set of commitments is closed under all three composition operators.

**Proof**: Each composition operator produces a valid 5-tuple with:
- Well-defined trigger (conjunction, disjunction, or inheritance)
- Well-defined action (sequencing, parallel execution, or branching)
- Well-defined verification (conjunction of component verifications)
- Valid failure set (union of component failures)
- Valid stake (sum or max of component stakes)

### Theorem 2 (Associativity of Sequential Composition)
$$(C_1 ; C_2) ; C_3 = C_1 ; (C_2 ; C_3)$$

**Proof**: Both sides produce the same action sequence $a_1 \to a_2 \to a_3$ and require all three verifications to succeed.

### Theorem 3 (Commutativity of Parallel Composition)
$$C_1 \| C_2 = C_2 \| C_1$$

**Proof**: Parallel execution is order-independent; both actions execute simultaneously.

### Theorem 4 (Determinism)
For any commitment $C$ and state transition $(s, s', a)$:
$$\text{VERIFY}(C, s, s', a) = \text{VERIFY}(C, s, s', a)$$

**Proof**: All predicates are pure functions; no randomness or side effects.

---

## A.5 Confidence Dynamics

### Definition 4 (Confidence)
Agent confidence in commitment $C$ after $n$ observations is:

$$\kappa_C = \frac{\alpha + \text{successes}}{\alpha + \beta + n}$$

where $\alpha, \beta$ are Laplace smoothing parameters (default: $\alpha = \beta = 1$).

**Properties**:
- Initial confidence: $\kappa_C^{(0)} = \frac{\alpha}{\alpha + \beta} = 0.5$
- Converges to empirical success rate as $n \to \infty$
- Bounded: $\kappa_C \in (0, 1)$ for all $n$

### Theorem 5 (Confidence Convergence)
$$\lim_{n \to \infty} \kappa_C = \frac{\text{successes}}{n} = p_{true}$$

where $p_{true}$ is the true success probability.

---

## A.6 Implementation Reference

The calculus is implemented in [`src/gcl/core/calculus.py`](../src/gcl/core/calculus.py) with:

| Formal Concept | Python Implementation |
|----------------|----------------------|
| Commitment $C$ | `Commitment[S, A]` dataclass |
| Failure mode $(f, s, d)$ | `FailureMode` dataclass |
| Outcome $O$ | `CommitmentOutcome` dataclass |
| ISSUE | `issue(commitment, issuer)` |
| VERIFY | `verify(commitment, pre, post, action)` |
| SETTLE | `settle(commitment, outcome)` |
| Sequential (;) | `sequential(c1, c2)` |
| Parallel (‖) | `parallel(c1, c2)` |
| Conditional (◁p▷) | `conditional(c1, c2, predicate)` |
| Confidence $\kappa$ | `confidence(successes, failures, alpha, beta)` |

**Test coverage**: 20 tests in [`tests/test_calculus.py`](../tests/test_calculus.py) verify all operations and theorems.

---

## A.7 Relationship to GCL Framework

This minimal calculus provides the **formal foundation** for the full GCL framework:

1. **Level 1 (Grounded Commitments)**: Direct implementation of this calculus
2. **Level 2 (Template Hierarchy)**: Parameterized commitment generators
3. **Level 3 (Commitment-Grounded RL)**: Learning over commitment selection
4. **Level 4 (Multi-Agent Coordination)**: Shared commitment verification

The calculus is intentionally minimal—5 primitives, 3 operations, 3 compositions—to enable formal analysis while remaining practically implementable.
