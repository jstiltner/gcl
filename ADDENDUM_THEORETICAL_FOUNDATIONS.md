# GCL Project Addendum: Theoretical Foundations & Expanded Scope

## IMPORTANT: Integration Note

This addendum expands the original GCL project prompt significantly. The core implementation work you've begun remains valid—these additions provide:

1. **Mathematical formalization** of the concepts you're implementing
2. **Theoretical results** to prove (making this a theory+empirics paper)
3. **Language emergence connection** (Option B from our planning)
4. **Expanded alignment framing** (Option D)

**Do not restart.** Continue building the core abstractions. This addendum informs the paper structure and adds theoretical sections alongside the empirical work.

---

## Expanded Paper Vision

The paper is now **"Grounded Commitment Learning: A Theory of Verifiable Coordination"** with four integrated contributions:

| Component | Type | Section |
|-----------|------|---------|
| GCL Framework | Conceptual | §2 |
| Theoretical Guarantees (Option A) | Formal | §3 |
| Language Emergence (Option B) | Theoretical | §4 |
| Empirical Validation (Option C) | Experimental | §5 |
| Alignment Implications (Option D) | Discussion | §6 |

This is a **full theory paper**: conceptual framework + formal results + empirical validation + implications.

---

## Part I: Mathematical Formalization

### Core Definitions

**Definition 1 (Grounded Commitment).** A grounded commitment is a tuple $C = (s, a, \phi, F, \sigma, \kappa)$ where:
- $s \in \mathcal{S}$ is the triggering state
- $a \in \mathcal{A}$ is the promised action
- $\phi: \mathcal{S} \times \mathcal{S} \times \mathcal{A} \to \{0, 1\}$ is the verification function
- $F = \{(f_i, c_i, \rho_i)\}_{i=1}^{k}$ is the set of failure modes, where $f_i$ is a predicate, $c_i$ is the consequence, and $\rho_i \in [0,1]$ is severity
- $\sigma \in \mathbb{R}^+$ is the stake
- $\kappa \in [0,1]$ is the confidence

**Definition 2 (Commitment Outcome).** Given commitment $C$ executed from pre-state $s$ resulting in post-state $s'$ via action $a$, the outcome is:

```
O(C, s, s', a) = 
    SUCCESS           if φ(s, s', a) = 1
    FAILURE_i         if φ(s, s', a) = 0 ∧ f_i(s, s', a) = 1 ∧ ∀_{j<i} f_j(s, s', a) = 0
```

Failure modes are evaluated in severity order; the first matching failure mode determines the outcome.

**Definition 3 (Commitment Reward).** The reward for commitment $C$ with outcome $O$ is:

```
R(C, O) = 
    σ · α + V(s')           if O = SUCCESS
    -σ · ρ_i + r_i          if O = FAILURE_i
```

where α is the success bonus rate, V(s') is the downstream value, and r_i is any remediation value for failure mode i.

### Implementation Note

These definitions map directly to your code:

```python
# Definition 1 → commitment.py
@dataclass
class GroundedCommitment:
    triggering_state: State                           # s
    promised_action: ActionSpec                       # a
    verification_fn: Callable[[State, State, Action], bool]  # φ
    failure_modes: List[FailureMode]                  # F
    stake: float                                      # σ
    confidence: float                                 # κ

# Definition 2 → verification.py
class VerificationEngine:
    def evaluate(self, commitment, pre_state, post_state, action) -> Outcome:
        if commitment.verification_fn(pre_state, post_state, action):
            return Outcome.SUCCESS
        for i, fm in enumerate(commitment.failure_modes):
            if fm.condition(pre_state, post_state, action):
                return Outcome.failure(i, fm)
        return Outcome.UNSPECIFIED_FAILURE

# Definition 3 → reward.py
def compute_reward(commitment: GroundedCommitment, outcome: Outcome) -> float:
    if outcome.is_success:
        return commitment.stake * ALPHA + outcome.downstream_value
    else:
        fm = outcome.failure_mode
        return -commitment.stake * fm.severity + fm.remediation_value
```

---

## Part II: Theoretical Results (Option A)

These are the theorems we will prove in the paper. Some require formal proof; others are proven by construction/demonstration.

### Communication Complexity

**Definition 4 (Semantic Space).** An agent $A$ has semantic space $\mathcal{E}_A \subseteq \mathbb{R}^d$, a d-dimensional embedding space. The interpretation function $I_A: \mathcal{M} \to \mathcal{E}_A$ maps messages to embeddings.

**Definition 5 (Semantic Drift).** For agents A and B, semantic drift is:

```
ε_AB = 𝔼_{m ~ ℳ}[‖I_A(m) - T_{A→B}(I_A(m))‖_2]
```

where $T_{A \to B}: \mathcal{E}_A \to \mathcal{E}_B$ is the optimal transport map between semantic spaces.

**Lemma 1 (Verification Invariance).** Commitment verification is invariant to semantic drift:

```
φ(s, s', a) = φ(s, s', a)   ∀ ε_AB
```

*Proof.* Verification φ operates on states and actions, not on representations. The function φ: S × S × A → {0,1} has no dependence on E_A or E_B. ∎

**Lemma 2 (Interpretation Error).** For representation-based communication, expected interpretation error scales with semantic drift:

```
𝔼[‖I_B(m) - I_A(m)‖] ≥ ε_AB
```

*Proof.* By definition of semantic drift and triangle inequality. ∎

**Theorem 1 (Communication Complexity).** Consider a task with n subtasks requiring coordination among k agents.

(a) Commitment-based coordination requires O(n · k) verification operations.

(b) Representation-based coordination requires O(n · k²) interpretation operations.

(c) The error rate for representation-based coordination scales as O(ε · n · k²) where ε = max_{i,j} ε_ij.

*Proof.*

(a) Each subtask requires at most k commitments (one per agent involved). Each commitment requires one verification. Total: O(n · k).

(b) Each subtask requires communication among k agents. Each pair must interpret each other's messages: C(k,2) = O(k²) interpretations per subtask. Total: O(n · k²).

(c) Each interpretation has error probability proportional to ε (by Lemma 2). With O(n · k²) interpretations, expected errors scale as O(ε · n · k²). ∎

**Corollary 1 (Drift Threshold).** There exists ε* > 0 such that for all ε > ε*, commitment-based coordination achieves higher task success than representation-based coordination.

*Proof.* As ε → ∞, interpretation error approaches 1 (random). Commitment verification remains exact (Lemma 1). ∎

### Experimental Validation for Theorem 1

```python
# experiments/03_multiagent_coordination.py should validate this

def experiment_communication_complexity():
    """
    Vary:
    - n (number of subtasks): [5, 10, 20, 50]
    - k (number of agents): [2, 3, 5, 8]
    - ε (semantic drift): [0, 0.1, 0.3, 0.5, 1.0]
    
    Measure:
    - Task success rate (commitment vs representation)
    - Communication overhead (messages/verifications)
    - Error rate
    
    Expected: Commitment dominates as ε and k increase
    """
    pass
```

### Convergence Results

**Definition 6 (Commitment MDP).** A commitment MDP is a tuple (S^C, C, P, R, γ) where:
- S^C = S × ℝ × 2^{C_active} is the commitment state space (environment state, reputation, active commitments)
- C is the space of possible commitments
- P: S^C × C → Δ(S^C) is the transition function
- R: S^C × C × S^C → ℝ is the reward function
- γ ∈ [0,1) is the discount factor

**Definition 7 (Commitment Policy).** A commitment policy π: S^C → Δ(C) maps commitment states to distributions over commitments.

**Assumption 1 (Verifiable Task Structure).** A task is *verifiable* if:
1. Success criteria are computable: ∃ algorithm to evaluate φ(s, s', a)
2. Failure modes are enumerable: |F| < ∞
3. Outcomes are observable: agent observes (s, s', a, O)

**Theorem 2 (Convergence).** Under Assumption 1, for any commitment MDP with finite state and commitment spaces, policy gradient methods converge to a locally optimal commitment policy π*.

*Proof sketch.*
1. The commitment MDP satisfies standard MDP assumptions (Markov property, bounded rewards under Assumption 1).
2. Verification provides unbiased reward signal (no hidden confounders).
3. Standard policy gradient convergence results apply (Sutton et al., 2000).
4. Local optimality follows from smoothness of policy parameterization. ∎

**Theorem 3 (Multi-Agent Equilibrium).** In a multi-agent commitment market with k agents, if all agents use commitment policies optimized via self-play, the joint policy converges to a Nash equilibrium of the coordination game.

*Proof sketch.*
1. The commitment market is a stochastic game with verifiable payoffs.
2. Self-play with policy gradient converges to Nash in certain game classes (Bowling & Veloso, 2002).
3. Verifiable commitments eliminate payoff uncertainty—each agent knows the exact consequence of commitment fulfillment/violation.
4. This satisfies conditions for convergence in potential games. ∎

### Implementation Note

The convergence theorems justify your RL training loop:

```python
# learning/training.py

class CommitmentPolicyTrainer:
    """
    Train commitment policy via PPO.
    
    Theorem 2 guarantees convergence to local optimum.
    Theorem 3 guarantees Nash equilibrium in multi-agent setting.
    
    Key requirements (Assumption 1):
    - verification_fn must be computable
    - failure_modes must be finite
    - outcomes must be observable
    """
    
    def train(self, env: CommitmentEnvironment, policy: CommitmentPolicy):
        # Standard PPO, but action space is commitments
        # Reward comes from verification (clean signal)
        pass
```

---

## Part III: Template Algebra

### Formal Definitions

**Definition 8 (Commitment Template).** A template is a tuple T = (τ, α, ψ, Θ) where:
- τ: S → {0, 1} is the trigger schema (when does this template apply?)
- α: S → A is the action schema (parameterized by state)
- ψ: S × S × A → {0, 1} is the verification schema
- Θ ⊆ S is the validated context region

**Definition 9 (Template Instantiation).** Given template T and state s ∈ Θ, instantiation produces commitment:

```
Instantiate(T, s) = (s, α(s), ψ, F_T, σ_T(s), κ_T(s))
```

where F_T is inherited failure modes, σ_T(s) is context-dependent stake, and κ_T(s) is context-dependent confidence.

**Definition 10 (Template Similarity).** For templates T_1, T_2, structural similarity is:

```
Sim(T_1, T_2) = λ_τ · Sim_τ(τ_1, τ_2) + λ_α · Sim_α(α_1, α_2) + λ_ψ · Sim_ψ(ψ_1, ψ_2)
```

where Sim_τ, Sim_α, Sim_ψ are similarity functions on schemas and λ_τ + λ_α + λ_ψ = 1.

### Template Composition Operators

**Definition 11 (Template Composition).** Templates compose via operators:

**Sequential composition:** T_1 ; T_2 executes T_1 then T_2:
```
τ_{T_1;T_2}(s) = τ_1(s)
α_{T_1;T_2}(s) = α_2(result(α_1(s)))
ψ_{T_1;T_2}(s, s', a) = ψ_1(s, s_mid, a_1) ∧ ψ_2(s_mid, s', a_2)
```

**Parallel composition:** T_1 ‖ T_2 executes both simultaneously:
```
τ_{T_1‖T_2}(s) = τ_1(s) ∧ τ_2(s)
α_{T_1‖T_2}(s) = (α_1(s), α_2(s))
ψ_{T_1‖T_2}(s, s', a) = ψ_1(s, s', a_1) ∧ ψ_2(s, s', a_2)
```

**Conditional composition:** T_1 ◁p▷ T_2 chooses based on predicate p:
```
τ_{T_1◁p▷T_2}(s) = τ_1(s) ∨ τ_2(s)
α_{T_1◁p▷T_2}(s) = α_1(s) if p(s) else α_2(s)
```

**Theorem 4 (Template Closure).** The set of templates is closed under sequential, parallel, and conditional composition.

*Proof.* Verify that composed structures satisfy Definition 8. Trigger, action, and verification schemas remain well-defined under each operator. ∎

### Implementation

```python
# templates/composition.py

class TemplateComposer:
    """Implements template algebra from Definition 11."""
    
    @staticmethod
    def sequential(t1: CommitmentTemplate, t2: CommitmentTemplate) -> CommitmentTemplate:
        """T_1 ; T_2: Execute t1, then t2 on result."""
        return CommitmentTemplate(
            trigger_schema=t1.trigger_schema,
            action_schema=lambda s: t2.action_schema(t1.execute(s)),
            verification_schema=lambda s, s_, a: (
                t1.verification_schema(s, t1.result_state, t1.action) and
                t2.verification_schema(t1.result_state, s_, a)
            ),
            validated_contexts=t1.validated_contexts & t2.validated_contexts
        )
    
    @staticmethod
    def parallel(t1: CommitmentTemplate, t2: CommitmentTemplate) -> CommitmentTemplate:
        """T_1 ‖ T_2: Execute both simultaneously."""
        return CommitmentTemplate(
            trigger_schema=lambda s: t1.trigger_schema(s) and t2.trigger_schema(s),
            action_schema=lambda s: (t1.action_schema(s), t2.action_schema(s)),
            verification_schema=lambda s, s_, a: (
                t1.verification_schema(s, s_, a[0]) and
                t2.verification_schema(s, s_, a[1])
            ),
            validated_contexts=t1.validated_contexts & t2.validated_contexts
        )
    
    @staticmethod
    def conditional(t1: CommitmentTemplate, t2: CommitmentTemplate, 
                    predicate: Callable[[State], bool]) -> CommitmentTemplate:
        """T_1 ◁p▷ T_2: Choose based on predicate."""
        return CommitmentTemplate(
            trigger_schema=lambda s: t1.trigger_schema(s) or t2.trigger_schema(s),
            action_schema=lambda s: t1.action_schema(s) if predicate(s) else t2.action_schema(s),
            verification_schema=lambda s, s_, a: (
                t1.verification_schema(s, s_, a) if predicate(s) 
                else t2.verification_schema(s, s_, a)
            ),
            validated_contexts=t1.validated_contexts | t2.validated_contexts
        )
```

### Analogical Transfer

**Theorem 5 (Analogical Transfer Bound).** Given template T validated in context region Θ and new context s' ∉ Θ, if ∃s ∈ Θ such that Sim_context(s, s') > θ, then:

```
𝔼[Success(Instantiate(T, s'))] ≥ κ_T(s) · Sim_context(s, s')
```

*Proof sketch.*
1. Template success in Θ implies the structure (τ, α, ψ) captures task-relevant features.
2. Context similarity implies shared structure between s and s'.
3. Success probability degrades gracefully with similarity (Lipschitz assumption on task structure).
4. Bound follows from confidence calibration: κ_T estimates success probability in validated contexts. ∎

**Experimental Validation:**

```python
# experiments/02_template_induction.py

def experiment_analogical_transfer():
    """
    1. Train agent on domain A, induce templates
    2. Transfer to domain B (varying similarity to A)
    3. Measure success rate vs Sim(A, B)
    
    Expected: Success rate ≥ confidence × similarity (Theorem 5)
    """
    pass
```

---

## Part IV: Commitment Semantics for Language (Option B)

This section connects GCL to natural language, arguing that language is grounded in commitment structure.

### Core Hypothesis

**Definition 12 (Commitment Semantics).** The meaning of utterance u is the commitment structure it instantiates:

```
⟦u⟧ = {C : Speaker commits to C by uttering u}
```

**Definition 13 (Illocutionary Commitment).** Following speech act theory, utterance types map to commitment types:

| Utterance Type | Commitment Structure |
|----------------|---------------------|
| Assertion "P" | C_assert = (⊤, claim(P), φ_truth, F_retraction, σ_rep, κ_belief) |
| Promise "I will X" | C_promise = (⊤, X, φ_completion, F_breach, σ_trust, κ_ability) |
| Request "Please X" | C_request = (⊤, await(X), φ_response, F_ignore, σ_social, κ_compliance) |
| Question "Is P?" | C_question = (⊤, await(answer(P)), φ_answered, F_unanswered, σ_attention, κ_response) |

### Gricean Maxims as Meta-Commitments

**Proposition 1 (Gricean Maxims as Commitments).** Grice's cooperative principle can be formalized as meta-commitments:

- **Quality:** C_quality = (speaking, assert(P), φ_believed, F_deception, σ_high, κ_sincere)
  "I commit that what I say, I believe to be true"

- **Quantity:** C_quantity = (speaking, inform, φ_{sufficient ∧ ¬excessive}, F_{under/over}, σ_med, κ_calibrated)
  "I commit to saying enough but not too much"

- **Relevance:** C_relevance = (speaking, contribute, φ_on-topic, F_tangent, σ_med, κ_focused)
  "I commit to staying on topic"

- **Manner:** C_manner = (speaking, express, φ_clear, F_obscure, σ_low, κ_articulate)
  "I commit to being clear"

### The Compression Hypothesis

**Hypothesis 1 (Commitment Compression).** Natural language grammar is a compression scheme for commitment templates:

```
Grammar ≈ Compress(𝒯_human)
```

where 𝒯_human is the set of commitment templates used in human coordination.

**Evidence:**
1. Grammatical constructions map systematically to commitment types
2. Emergent communication develops commitment-like structure before grammar
3. Children acquire speech acts before full syntax
4. Cross-linguistic universals reflect universal coordination needs

### Connection to Emergent Communication

This hypothesis predicts that emergent communication systems should develop commitment-like structure. This is testable:

```python
# experiments/06_emergent_language.py

def experiment_emergent_commitment_structure():
    """
    1. Train agents to coordinate via learned protocol (no English)
    2. Analyze emergent protocol structure
    3. Test whether messages function as commitments:
       - Do messages predict future behavior?
       - Do violations have consequences?
       - Do patterns abstract into templates?
    
    Prediction: Emergent protocols will have commitment structure
    even without being trained for it.
    """
    pass
```

### Paper Section Structure

For the language section (§4), structure as:

1. **Commitment Semantics Hypothesis** — meaning is commitment structure
2. **Formalization** — Definitions 12-13, Proposition 1
3. **Relationship to Speech Act Theory** — show GCL formalizes Austin/Searle
4. **Relationship to Grice** — show maxims are meta-commitments
5. **Predictions for Emergent Communication** — testable claims
6. **Discussion** — we don't prove this, but show it's productive

---

## Part V: Alignment as Commitment Management (Option D)

### Core Argument

Current alignment approaches assume we need AI systems to "understand" human values. But:
- Understanding is unverifiable (we can't inspect internal states)
- Understanding is unstable (fine-tuning can shift it)
- Understanding doesn't guarantee behavior (knowing ≠ doing)

GCL reframes alignment: **aligned AI = AI that makes and keeps value-consistent commitments**.

### Formal Definitions

**Definition 14 (Value-Aligned Commitment).** A commitment C is aligned with value function V_human: S → ℝ if:

```
𝔼[V_human(s') | C fulfilled] ≥ 𝔼[V_human(s') | C not made]
```

**Definition 15 (Commitment-Grounded Alignment).** An agent is commitment-aligned if:
1. It makes only value-aligned commitments
2. It fulfills its commitments with probability ≥ 1 - δ
3. Its commitment policy improves over time: d/dt 𝔼[V_human] ≥ 0

### The Verifiability Theorem

**Theorem 6 (Alignment Verifiability).** Commitment-grounded alignment is verifiable, while representation-grounded alignment is not.

*Proof.*

(a) For commitment-grounded alignment:
- Condition 1 is verifiable: evaluate V_human on commitment outcomes
- Condition 2 is verifiable: track fulfillment rate
- Condition 3 is verifiable: measure V_human over time

(b) For representation-grounded alignment ("agent understands values"):
- "Understanding" requires access to internal representations
- Internal representations are not directly observable
- Any behavioral test can be gamed by a sufficiently capable misaligned agent (deceptive alignment)

Therefore, commitment-grounded alignment is verifiable while representation-grounded alignment is not. ∎

### Decomposition

**Corollary 2 (Commitment Alignment Decomposition).** Complex value alignment decomposes into simple commitment verification:

```
Aligned(V_complex) ⟺ ⋀_{i=1}^{n} Fulfilled(C_i)
```

where {C_i} is a commitment portfolio that collectively satisfies V_complex.

### Implications for AI Safety

This reframing has practical implications:

1. **Constitutional AI as Commitments:** Anthropic's constitutional AI can be viewed as training models to make and keep commitments to constitutional principles.

2. **RLHF as Commitment Learning:** RLHF trains models to make commitments humans endorse (via preference signal).

3. **Interpretability via Commitments:** Instead of interpreting internal states, audit commitment portfolios.

4. **Robustness via Verification:** Even if internal "values" drift, commitment verification catches violations.

### Experimental Demonstration

```python
# experiments/05_alignment_demo.py

def experiment_alignment_verification():
    """
    Demonstrate that commitment-based alignment is verifiable.
    
    Setup:
    1. Define value function V_human (e.g., helpfulness + harmlessness)
    2. Train agent with commitment-based alignment
    3. Train agent with representation-based alignment (standard RLHF)
    
    Test:
    - Introduce distribution shift
    - Measure alignment maintenance
    - Commitment-based: can verify via commitment auditing
    - Representation-based: cannot verify (must trust internal state)
    
    Expected: Commitment-based alignment is auditable; representation-based is not.
    """
    pass
```

---

## Part VI: Summary of Theoretical Contributions

| Result | Type | Section | Difficulty | Status |
|--------|------|---------|------------|--------|
| Theorem 1 (Communication Complexity) | Complexity bound | §3.1 | Medium | To prove |
| Corollary 1 (Drift Threshold) | Robustness | §3.1 | Easy | To prove |
| Theorem 2 (Convergence) | RL theory | §3.2 | Medium | Proof sketch done |
| Theorem 3 (Multi-Agent Equilibrium) | Game theory | §3.2 | Medium | Proof sketch done |
| Theorem 4 (Template Closure) | Algebra | §3.3 | Easy | To prove |
| Theorem 5 (Analogical Transfer) | Transfer learning | §3.3 | Medium | To prove |
| Proposition 1 (Gricean Maxims) | Linguistics | §4 | Easy | Done |
| Hypothesis 1 (Compression) | Linguistics | §4 | N/A (hypothesis) | — |
| Theorem 6 (Alignment Verifiability) | Safety | §6 | Easy | Proof done |
| Corollary 2 (Decomposition) | Safety | §6 | Easy | Done |

---

## Part VII: Updated Paper Structure

```
Grounded Commitment Learning: A Theory of Verifiable Coordination

Abstract
- The interpretation problem in AI coordination
- GCL: commitments, not representations
- Key results: complexity, convergence, transfer, alignment

1. Introduction
   1.1 The Interpretation Problem
   1.2 GCL as Alternative
   1.3 Contributions

2. The GCL Framework
   2.1 Grounded Commitments (Definition 1-3)
   2.2 Failure-First Specification
   2.3 Template Hierarchy (Definition 8-11)
   2.4 Commitment-Grounded RL (Definition 6-7)
   2.5 Multi-Agent Commitment Markets

3. Theoretical Results
   3.1 Communication Complexity (Theorem 1, Corollary 1)
   3.2 Convergence and Equilibrium (Theorems 2-3)
   3.3 Template Algebra and Transfer (Theorems 4-5)

4. Connection to Natural Language
   4.1 Commitment Semantics (Definition 12-13)
   4.2 Gricean Maxims as Commitments (Proposition 1)
   4.3 The Compression Hypothesis
   4.4 Predictions for Emergent Communication

5. Experiments
   5.1 Single-Agent Commitment Learning
   5.2 Template Induction and Transfer
   5.3 Multi-Agent Coordination Emergence
   5.4 Heterogeneous LLM Coordination
   5.5 Emergent Language Structure (Optional)

6. Implications for AI Alignment
   6.1 The Problem with Understanding-Based Alignment
   6.2 Commitment-Grounded Alignment (Definitions 14-15)
   6.3 Verifiability (Theorem 6)
   6.4 Practical Implications

7. Related Work
   7.1 Mechanism Design and Contract Theory
   7.2 Formal Methods and Design by Contract
   7.3 Emergent Communication
   7.4 Speech Act Theory and Pragmatics
   7.5 AI Safety and Alignment

8. Conclusion

Appendices
   A. Full Proofs
   B. Experimental Details
   C. Implementation
```

---

## Part VIII: Updated Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Core abstractions | commitment.py, verification.py, reputation.py |
| 2 | Single-agent RL | environment.py, policy.py, training.py, Experiment 1 |
| 3 | Templates + Theory | template.py, composition.py, Proofs for Theorems 1-5 |
| 4 | Multi-agent | market.py, matching.py, Experiment 3-4 |
| 5 | LLM integration + Language | adapter.py, Experiment 5, Language section draft |
| 6 | Alignment + Writing | Full paper draft, Theorem 6 proof, Alignment section |
| 7 | Polish + Submit | arXiv submission, code cleanup, blog post |

---

## Part IX: Key Equations Quick Reference

For easy reference when writing the paper:

**Commitment Outcome:**
```
O(C, s, s', a) = SUCCESS if φ(s, s', a) = 1; FAILURE_i otherwise
```

**Reward:**
```
R(C, O) = σα + V(s') if success; -σρ_i + r_i if failure
```

**Semantic Drift:**
```
ε_AB = 𝔼_{m}[‖I_A(m) - T_{A→B}(I_A(m))‖]
```

**Communication Complexity:**
```
Commitment: O(n·k)    vs    Representation: O(n·k²)
Error rate: O(1)      vs    O(ε·n·k²)
```

**Template Similarity:**
```
Sim(T_1, T_2) = λ_τ·Sim_τ + λ_α·Sim_α + λ_ψ·Sim_ψ
```

**Analogical Transfer Bound:**
```
𝔼[Success] ≥ κ_T(s) · Sim_context(s, s')
```

**Alignment Verifiability:**
```
Commitment-aligned ⟺ (value-aligned commits) ∧ (fulfillment ≥ 1-δ) ∧ (improving)
All three conditions are verifiable; "understanding" is not.
```

---

## Final Notes

This addendum transforms GCL from a solid engineering contribution into a **full theoretical framework**. You now have:

1. **Formal definitions** that ground the implementation
2. **Theorems** that establish guarantees
3. **Language connection** that broadens impact
4. **Alignment framing** that speaks to Anthropic's mission

The implementation work remains the same—these additions inform the paper, not the code (except for experimental validation of the theorems).

**Execute the implementation. The theory is ready.**