# GCL Project Addendum 2: Population Dynamics & Experimental Rigor

## IMPORTANT: Integration Note

This is the second addendum to the GCL project. It covers:

1. **Population-scale experiments** — testing emergence at 50-500 agents
2. **Lightweight agent architecture** — for local compute feasibility
3. **Rigorous experimental design** — baselines, statistics, falsifiable predictions
4. **Institutional emergence narrative** — connecting to human coordination phenomena
5. **Compute strategy** — what runs locally vs. API

**The key insight:** We need two experimental tracks:
- **LLM experiments** (2-8 agents, API) → validates "this works with real AI"
- **Population experiments** (50-500 agents, local) → validates "this exhibits emergent dynamics"

Both are essential. Neither alone is sufficient.

---

## Part I: The Two-Track Experimental Strategy

### Track 1: LLM Validation (API)

**Purpose:** Demonstrate GCL works with real AI systems people care about.

**Agents:** Claude, GPT-4, Llama, Mistral — actual LLMs via API

**Scale:** 2-8 agents (API cost constraints)

**Key experiments:**
- Head-to-head vs. natural language (Experiment A)
- Drift robustness across model families (Experiment B)
- Heterogeneous coordination (Claude + GPT) (Experiment D from original)

**Compute:** API calls, ~$500-1000 total

**Outcome:** "GCL enables better coordination between real LLMs"

### Track 2: Population Dynamics (Local)

**Purpose:** Demonstrate GCL exhibits emergent coordination phenomena at scale.

**Agents:** Lightweight neural network policies (not LLMs)

**Scale:** 50-500 agents

**Key experiments:**
- Protocol convergence (Experiment H.1)
- Trust network formation (Experiment H.2)
- Template propagation (Experiment H.3)
- Specialization emergence (Experiment H.4)
- Critical population thresholds (Experiment H.5)

**Compute:** Local GPU/CPU, hours not days

**Outcome:** "GCL reproduces dynamics seen in human institutions"

### Why Both Tracks?

| Question | Track 1 Answers | Track 2 Answers |
|----------|-----------------|-----------------|
| Does this work with real AI? | ✅ | ❌ |
| Does this scale? | ❌ (too expensive) | ✅ |
| Are there emergent phenomena? | ❌ (too few agents) | ✅ |
| Is this practical? | ✅ | ⚠️ (simplified agents) |
| Does this match theory? | ✅ (Theorems 1-3) | ✅ (Predictions 1-5) |

**Together:** "GCL works with real LLMs AND exhibits population-level emergence"

---

## Part II: Lightweight Agent Architecture

For population experiments, we need agents that:
- Run fast (thousands of interactions per second)
- Learn commitment policies
- Maintain reputation and templates
- Don't require API calls

### Core Architecture

```python
# src/gcl/population/lightweight_agent.py

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class AgentConfig:
    """Configuration for lightweight commitment agent."""
    state_dim: int = 32          # Environment state dimension
    hidden_dim: int = 64         # Hidden layer size
    commitment_vocab: int = 16   # Number of commitment types
    template_capacity: int = 10  # Max templates to store
    learning_rate: float = 1e-3
    reputation_decay: float = 0.99  # Reputation decay per timestep
    

class CommitmentEncoder(nn.Module):
    """Encode state into commitment logits."""
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        # Output heads
        self.type_head = nn.Linear(config.hidden_dim, config.commitment_vocab)
        self.confidence_head = nn.Linear(config.hidden_dim, 1)
        self.stake_head = nn.Linear(config.hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(state)
        return {
            "type_logits": self.type_head(h),
            "confidence": torch.sigmoid(self.confidence_head(h)),
            "stake": torch.softplus(self.stake_head(h)),
        }


class LightweightCommitmentAgent:
    """
    Fast, lightweight agent for population experiments.
    
    Not an LLM — a small learned policy that:
    - Proposes commitments based on state
    - Tracks its own reputation
    - Maintains a library of learned templates
    - Updates policy via simple RL
    """
    
    def __init__(self, agent_id: str, config: AgentConfig = None):
        self.agent_id = agent_id
        self.config = config or AgentConfig()
        
        # Policy network
        self.encoder = CommitmentEncoder(self.config)
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(), 
            lr=self.config.learning_rate
        )
        
        # State
        self.reputation = 1.0
        self.templates: List[LearnedTemplate] = []
        self.commitment_history: List[CommitmentRecord] = []
        
        # Statistics
        self.total_commitments = 0
        self.fulfilled_commitments = 0
        self.stats = AgentStats()
    
    def propose_commitment(
        self, 
        state: torch.Tensor,
        partner_reputation: float = 1.0
    ) -> "LightweightCommitment":
        """
        Propose a commitment given current state.
        
        The agent considers:
        - Current state (what needs to be done)
        - Own reputation (how much can I stake?)
        - Partner reputation (how much should I trust them?)
        - Available templates (what patterns have worked?)
        """
        with torch.no_grad():
            outputs = self.encoder(state)
        
        # Sample commitment type
        type_probs = torch.softmax(outputs["type_logits"], dim=-1)
        commitment_type = torch.multinomial(type_probs, 1).item()
        
        # Determine confidence and stake
        confidence = outputs["confidence"].item()
        base_stake = outputs["stake"].item()
        
        # Modulate stake by reputation (can't stake more than you have)
        actual_stake = min(base_stake, self.reputation * 0.5)
        
        # Check if we have a relevant template
        template = self._find_matching_template(state, commitment_type)
        
        return LightweightCommitment(
            issuer=self.agent_id,
            commitment_type=commitment_type,
            confidence=confidence,
            stake=actual_stake,
            template_id=template.id if template else None,
            state_hash=self._hash_state(state)
        )
    
    def receive_outcome(
        self, 
        commitment: "LightweightCommitment",
        outcome: "CommitmentOutcome",
        reward: float
    ):
        """Update agent based on commitment outcome."""
        
        # Update reputation
        if outcome.success:
            self.reputation = min(2.0, self.reputation + 0.1 * commitment.stake)
            self.fulfilled_commitments += 1
        else:
            self.reputation = max(0.1, self.reputation - outcome.severity * commitment.stake)
        
        self.reputation *= self.config.reputation_decay  # Natural decay
        self.total_commitments += 1
        
        # Record for template learning
        self.commitment_history.append(CommitmentRecord(
            commitment=commitment,
            outcome=outcome,
            reward=reward
        ))
        
        # Update stats
        self.stats.update(commitment, outcome)
        
        # Maybe induce new template
        if len(self.commitment_history) % 50 == 0:
            self._maybe_induce_template()
    
    def update_policy(self, batch: List["CommitmentRecord"]):
        """
        Policy gradient update from batch of experiences.
        
        Simple REINFORCE with baseline.
        """
        if len(batch) < 10:
            return
        
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        baseline = np.mean([r.reward for r in batch])
        
        for record in batch:
            state = record.commitment.state_tensor
            outputs = self.encoder(state)
            
            # Log probability of taken action
            type_logits = outputs["type_logits"]
            log_prob = torch.log_softmax(type_logits, dim=-1)[record.commitment.commitment_type]
            
            # Policy gradient
            advantage = record.reward - baseline
            loss = -log_prob * advantage
            total_loss += loss
        
        total_loss /= len(batch)
        total_loss.backward()
        self.optimizer.step()
    
    def _find_matching_template(
        self, 
        state: torch.Tensor, 
        commitment_type: int
    ) -> Optional["LearnedTemplate"]:
        """Find a template that matches current context."""
        best_match = None
        best_score = 0.0
        
        for template in self.templates:
            if template.commitment_type != commitment_type:
                continue
            score = template.match_score(state)
            if score > best_score and score > 0.5:  # Threshold
                best_match = template
                best_score = score
        
        return best_match
    
    def _maybe_induce_template(self):
        """
        Induce a new template from recent successful commitments.
        
        Look for patterns: similar states + same commitment type + success
        """
        recent = self.commitment_history[-100:]
        successes = [r for r in recent if r.outcome.success]
        
        if len(successes) < 10:
            return
        
        # Cluster by commitment type
        by_type = {}
        for s in successes:
            t = s.commitment.commitment_type
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(s)
        
        # For each type with enough examples, maybe create template
        for commit_type, records in by_type.items():
            if len(records) < 5:
                continue
            
            # Check if states are similar (potential template)
            states = [r.commitment.state_tensor for r in records]
            state_var = torch.var(torch.stack(states), dim=0).mean().item()
            
            if state_var < 0.5:  # States are similar enough
                # Induce template
                template = LearnedTemplate(
                    id=f"{self.agent_id}_t{len(self.templates)}",
                    commitment_type=commit_type,
                    prototype_state=torch.mean(torch.stack(states), dim=0),
                    success_rate=len(records) / len([r for r in recent if r.commitment.commitment_type == commit_type]),
                    context_variance=state_var
                )
                
                # Add if we have capacity, or replace worst
                if len(self.templates) < self.config.template_capacity:
                    self.templates.append(template)
                else:
                    worst = min(self.templates, key=lambda t: t.success_rate)
                    if template.success_rate > worst.success_rate:
                        self.templates.remove(worst)
                        self.templates.append(template)
    
    def _hash_state(self, state: torch.Tensor) -> int:
        """Simple state hash for commitment identification."""
        return hash(tuple(state.numpy().round(2).flatten().tolist()))


@dataclass
class LightweightCommitment:
    """Simplified commitment for population experiments."""
    issuer: str
    commitment_type: int
    confidence: float
    stake: float
    template_id: Optional[str] = None
    state_hash: int = 0
    state_tensor: torch.Tensor = None  # Set during creation


@dataclass
class CommitmentOutcome:
    """Outcome of a commitment."""
    success: bool
    severity: float = 0.0  # If failure, how bad (0-1)
    failure_type: Optional[int] = None


@dataclass
class CommitmentRecord:
    """Record of commitment + outcome for learning."""
    commitment: LightweightCommitment
    outcome: CommitmentOutcome
    reward: float


@dataclass
class LearnedTemplate:
    """Template induced from successful commitments."""
    id: str
    commitment_type: int
    prototype_state: torch.Tensor
    success_rate: float
    context_variance: float
    times_used: int = 0
    times_succeeded: int = 0
    
    def match_score(self, state: torch.Tensor) -> float:
        """How well does this state match the template?"""
        distance = torch.norm(state - self.prototype_state).item()
        # Convert distance to similarity (0-1)
        return np.exp(-distance / (self.context_variance + 0.1))


@dataclass
class AgentStats:
    """Running statistics for analysis."""
    commitment_types_used: Dict[int, int] = field(default_factory=dict)
    success_by_type: Dict[int, List[bool]] = field(default_factory=dict)
    reputation_history: List[float] = field(default_factory=list)
    template_usage: Dict[str, int] = field(default_factory=dict)
    
    def update(self, commitment: LightweightCommitment, outcome: CommitmentOutcome):
        t = commitment.commitment_type
        self.commitment_types_used[t] = self.commitment_types_used.get(t, 0) + 1
        if t not in self.success_by_type:
            self.success_by_type[t] = []
        self.success_by_type[t].append(outcome.success)
        if commitment.template_id:
            self.template_usage[commitment.template_id] = \
                self.template_usage.get(commitment.template_id, 0) + 1
```

### Population Environment

```python
# src/gcl/population/environment.py

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import random

@dataclass
class PopulationConfig:
    """Configuration for population experiment."""
    n_agents: int = 100
    n_timesteps: int = 10000
    task_complexity: int = 4        # Subtasks per interaction
    interaction_type: str = "random"  # "random", "network", "spatial"
    commitment_vocab: int = 16
    state_dim: int = 32


class PopulationEnvironment:
    """
    Environment for population-scale commitment experiments.
    
    Agents interact pairwise, exchanging commitments to complete tasks.
    No central coordination — all structure emerges from agent interactions.
    """
    
    def __init__(self, config: PopulationConfig):
        self.config = config
        
        # Create agents
        agent_config = AgentConfig(
            state_dim=config.state_dim,
            commitment_vocab=config.commitment_vocab
        )
        self.agents = [
            LightweightCommitmentAgent(f"agent_{i}", agent_config)
            for i in range(config.n_agents)
        ]
        
        # Trust network (who has interacted with whom, outcomes)
        self.trust_matrix = np.ones((config.n_agents, config.n_agents))
        
        # Metrics tracking
        self.metrics = PopulationMetrics()
        
        # Timestep counter
        self.timestep = 0
    
    def step(self) -> Dict:
        """
        Run one timestep of population dynamics.
        
        1. Select pairs of agents to interact
        2. Each pair attempts to coordinate via commitments
        3. Resolve outcomes, update reputations
        4. Agents learn from experience
        5. Record metrics
        """
        self.timestep += 1
        
        # Select interacting pairs
        pairs = self._select_pairs()
        
        timestep_outcomes = []
        
        for i, j in pairs:
            agent_i = self.agents[i]
            agent_j = self.agents[j]
            
            # Generate task
            task = self._generate_task()
            
            # Agents propose commitments
            state_i = self._get_agent_state(i, task)
            state_j = self._get_agent_state(j, task)
            
            commit_i = agent_i.propose_commitment(
                state_i, 
                partner_reputation=agent_j.reputation
            )
            commit_j = agent_j.propose_commitment(
                state_j,
                partner_reputation=agent_i.reputation
            )
            
            # Resolve task based on commitments
            outcome_i, outcome_j, task_success = self._resolve_task(
                task, commit_i, commit_j, agent_i, agent_j
            )
            
            # Compute rewards
            reward_i = self._compute_reward(commit_i, outcome_i, task_success)
            reward_j = self._compute_reward(commit_j, outcome_j, task_success)
            
            # Agents receive outcomes
            agent_i.receive_outcome(commit_i, outcome_i, reward_i)
            agent_j.receive_outcome(commit_j, outcome_j, reward_j)
            
            # Update trust matrix
            self._update_trust(i, j, outcome_i.success, outcome_j.success)
            
            timestep_outcomes.append({
                "pair": (i, j),
                "task_success": task_success,
                "commitments": (commit_i.commitment_type, commit_j.commitment_type),
                "outcomes": (outcome_i.success, outcome_j.success)
            })
        
        # Periodic policy updates
        if self.timestep % 10 == 0:
            for agent in self.agents:
                if len(agent.commitment_history) >= 20:
                    batch = agent.commitment_history[-20:]
                    agent.update_policy(batch)
        
        # Record metrics
        self.metrics.record_timestep(self, timestep_outcomes)
        
        return {
            "timestep": self.timestep,
            "outcomes": timestep_outcomes,
            "summary": self.metrics.get_summary()
        }
    
    def _select_pairs(self) -> List[Tuple[int, int]]:
        """Select agent pairs for this timestep."""
        n = self.config.n_agents
        
        if self.config.interaction_type == "random":
            # Random pairing
            indices = list(range(n))
            random.shuffle(indices)
            return [(indices[i], indices[i+1]) for i in range(0, n-1, 2)]
        
        elif self.config.interaction_type == "network":
            # Preferential interaction based on trust
            pairs = []
            available = set(range(n))
            while len(available) >= 2:
                i = random.choice(list(available))
                available.remove(i)
                # Select partner weighted by trust
                candidates = list(available)
                weights = [self.trust_matrix[i, j] for j in candidates]
                weights = np.array(weights) / sum(weights)
                j = np.random.choice(candidates, p=weights)
                available.remove(j)
                pairs.append((i, j))
            return pairs
        
        else:
            raise ValueError(f"Unknown interaction type: {self.config.interaction_type}")
    
    def _generate_task(self) -> "Task":
        """Generate a random task requiring coordination."""
        return Task(
            subtasks=self.config.task_complexity,
            state=torch.randn(self.config.state_dim),
            required_types=random.sample(
                range(self.config.commitment_vocab),
                k=min(self.config.task_complexity, self.config.commitment_vocab)
            )
        )
    
    def _get_agent_state(self, agent_idx: int, task: "Task") -> torch.Tensor:
        """Get state observation for an agent."""
        # Combine task state with agent's private info
        agent = self.agents[agent_idx]
        private_state = torch.tensor([
            agent.reputation,
            agent.total_commitments / 100.0,
            agent.fulfilled_commitments / max(1, agent.total_commitments),
            len(agent.templates) / agent.config.template_capacity
        ])
        return torch.cat([task.state, private_state.float()])
    
    def _resolve_task(
        self,
        task: "Task",
        commit_i: LightweightCommitment,
        commit_j: LightweightCommitment,
        agent_i: LightweightCommitmentAgent,
        agent_j: LightweightCommitmentAgent
    ) -> Tuple[CommitmentOutcome, CommitmentOutcome, bool]:
        """
        Resolve task based on commitments.
        
        Simple model:
        - Task needs certain commitment types
        - Agents succeed based on confidence calibration
        - Task succeeds if both agents fulfill commitments
        """
        # Check if commitment types are useful for task
        useful_i = commit_i.commitment_type in task.required_types
        useful_j = commit_j.commitment_type in task.required_types
        
        # Agent succeeds if:
        # - Commitment is useful for task
        # - Random roll beats (1 - confidence) — calibration check
        # - Using a template with good track record helps
        
        def agent_succeeds(commit, agent, useful):
            if not useful:
                return False, 0.8  # Wrong commitment type is a failure
            
            # Base success probability from confidence
            base_prob = commit.confidence
            
            # Template bonus
            if commit.template_id:
                template = next(
                    (t for t in agent.templates if t.id == commit.template_id), 
                    None
                )
                if template:
                    base_prob = 0.5 * base_prob + 0.5 * template.success_rate
            
            success = random.random() < base_prob
            severity = 0.5 if not success else 0.0
            return success, severity
        
        success_i, severity_i = agent_succeeds(commit_i, agent_i, useful_i)
        success_j, severity_j = agent_succeeds(commit_j, agent_j, useful_j)
        
        task_success = success_i and success_j
        
        return (
            CommitmentOutcome(success=success_i, severity=severity_i),
            CommitmentOutcome(success=success_j, severity=severity_j),
            task_success
        )
    
    def _compute_reward(
        self,
        commitment: LightweightCommitment,
        outcome: CommitmentOutcome,
        task_success: bool
    ) -> float:
        """Compute reward for an agent."""
        if outcome.success:
            reward = commitment.stake * 0.1  # Keep stake
            if task_success:
                reward += 1.0  # Task bonus
            return reward
        else:
            return -commitment.stake * outcome.severity
    
    def _update_trust(self, i: int, j: int, success_i: bool, success_j: bool):
        """Update trust matrix based on interaction outcome."""
        # Trust increases with successful interactions
        alpha = 0.1  # Learning rate for trust
        
        if success_j:  # i's trust in j
            self.trust_matrix[i, j] = (1 - alpha) * self.trust_matrix[i, j] + alpha * 1.5
        else:
            self.trust_matrix[i, j] = (1 - alpha) * self.trust_matrix[i, j] + alpha * 0.5
        
        if success_i:  # j's trust in i
            self.trust_matrix[j, i] = (1 - alpha) * self.trust_matrix[j, i] + alpha * 1.5
        else:
            self.trust_matrix[j, i] = (1 - alpha) * self.trust_matrix[j, i] + alpha * 0.5
        
        # Clip trust values
        self.trust_matrix = np.clip(self.trust_matrix, 0.1, 3.0)
    
    def run(self, n_steps: int = None) -> "PopulationMetrics":
        """Run simulation for n_steps."""
        n_steps = n_steps or self.config.n_timesteps
        for _ in range(n_steps):
            self.step()
        return self.metrics


@dataclass
class Task:
    """A coordination task requiring multiple commitment types."""
    subtasks: int
    state: torch.Tensor
    required_types: List[int]
```

### Population Metrics

```python
# src/gcl/population/metrics.py

import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field
from collections import Counter
import networkx as nx

@dataclass
class PopulationMetrics:
    """Track population-level metrics over time."""
    
    # Time series
    timesteps: List[int] = field(default_factory=list)
    task_success_rate: List[float] = field(default_factory=list)
    avg_reputation: List[float] = field(default_factory=list)
    reputation_gini: List[float] = field(default_factory=list)
    commitment_entropy: List[float] = field(default_factory=list)
    template_count: List[int] = field(default_factory=list)
    specialization_index: List[float] = field(default_factory=list)
    
    # Network metrics (computed periodically)
    trust_network_metrics: List[Dict] = field(default_factory=list)
    
    def record_timestep(self, env: "PopulationEnvironment", outcomes: List[Dict]):
        """Record metrics for one timestep."""
        self.timesteps.append(env.timestep)
        
        # Task success rate
        if outcomes:
            success_rate = np.mean([o["task_success"] for o in outcomes])
            self.task_success_rate.append(success_rate)
        
        # Reputation statistics
        reps = [a.reputation for a in env.agents]
        self.avg_reputation.append(np.mean(reps))
        self.reputation_gini.append(self._gini_coefficient(reps))
        
        # Commitment type entropy (protocol convergence)
        all_types = []
        for o in outcomes:
            all_types.extend(o["commitments"])
        if all_types:
            self.commitment_entropy.append(self._entropy(all_types))
        
        # Template statistics
        total_templates = sum(len(a.templates) for a in env.agents)
        self.template_count.append(total_templates)
        
        # Specialization index
        self.specialization_index.append(self._compute_specialization(env))
        
        # Network metrics (every 100 steps)
        if env.timestep % 100 == 0:
            self.trust_network_metrics.append(
                self._compute_network_metrics(env)
            )
    
    def _gini_coefficient(self, values: List[float]) -> float:
        """Compute Gini coefficient (0 = equal, 1 = unequal)."""
        values = np.array(sorted(values))
        n = len(values)
        if n == 0 or np.sum(values) == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))
    
    def _entropy(self, types: List[int]) -> float:
        """Compute entropy of commitment type distribution."""
        counts = Counter(types)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return -sum(p * np.log(p + 1e-10) for p in probs)
    
    def _compute_specialization(self, env: "PopulationEnvironment") -> float:
        """
        Compute specialization index.
        
        High = agents specialize in few commitment types
        Low = agents use many commitment types equally
        """
        agent_ginis = []
        for agent in env.agents:
            if agent.stats.commitment_types_used:
                counts = list(agent.stats.commitment_types_used.values())
                agent_ginis.append(self._gini_coefficient(counts))
        return np.mean(agent_ginis) if agent_ginis else 0.0
    
    def _compute_network_metrics(self, env: "PopulationEnvironment") -> Dict:
        """Compute trust network topology metrics."""
        # Build graph from trust matrix
        G = nx.DiGraph()
        n = len(env.agents)
        
        # Add edges where trust is above threshold
        threshold = 1.2  # Above baseline trust
        for i in range(n):
            for j in range(n):
                if i != j and env.trust_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=env.trust_matrix[i, j])
        
        if G.number_of_edges() == 0:
            return {"edges": 0}
        
        # Compute metrics
        metrics = {
            "edges": G.number_of_edges(),
            "density": nx.density(G),
        }
        
        # Clustering (on undirected version)
        G_undirected = G.to_undirected()
        if G_undirected.number_of_edges() > 0:
            metrics["clustering"] = nx.average_clustering(G_undirected)
            
            # Average path length (if connected)
            if nx.is_connected(G_undirected):
                metrics["avg_path_length"] = nx.average_shortest_path_length(G_undirected)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                subgraph = G_undirected.subgraph(largest_cc)
                metrics["avg_path_length"] = nx.average_shortest_path_length(subgraph)
                metrics["largest_cc_fraction"] = len(largest_cc) / n
        
        # Degree distribution
        degrees = [d for _, d in G.degree()]
        metrics["avg_degree"] = np.mean(degrees)
        metrics["degree_std"] = np.std(degrees)
        
        # PageRank (identifies influential agents)
        pagerank = nx.pagerank(G)
        metrics["pagerank_gini"] = self._gini_coefficient(list(pagerank.values()))
        
        return metrics
    
    def get_summary(self) -> Dict:
        """Get summary of current metrics."""
        return {
            "timestep": self.timesteps[-1] if self.timesteps else 0,
            "task_success_rate": self.task_success_rate[-1] if self.task_success_rate else 0,
            "avg_reputation": self.avg_reputation[-1] if self.avg_reputation else 1.0,
            "commitment_entropy": self.commitment_entropy[-1] if self.commitment_entropy else 0,
            "template_count": self.template_count[-1] if self.template_count else 0,
            "specialization": self.specialization_index[-1] if self.specialization_index else 0,
        }
    
    def get_convergence_analysis(self) -> Dict:
        """Analyze whether protocol has converged."""
        if len(self.commitment_entropy) < 100:
            return {"converged": False, "reason": "insufficient_data"}
        
        # Check if entropy is decreasing
        early_entropy = np.mean(self.commitment_entropy[:50])
        late_entropy = np.mean(self.commitment_entropy[-50:])
        
        entropy_decrease = (early_entropy - late_entropy) / (early_entropy + 1e-10)
        
        # Check if success rate is increasing
        early_success = np.mean(self.task_success_rate[:50])
        late_success = np.mean(self.task_success_rate[-50:])
        
        success_increase = late_success - early_success
        
        return {
            "converged": entropy_decrease > 0.2 and success_increase > 0.1,
            "entropy_decrease": entropy_decrease,
            "success_increase": success_increase,
            "early_entropy": early_entropy,
            "late_entropy": late_entropy,
            "early_success": early_success,
            "late_success": late_success
        }
```

---

## Part III: Falsifiable Predictions

These predictions are specific, quantitative, and testable. If they fail, the theory needs revision.

### Prediction 1: Protocol Convergence

**Statement:** In a population of N ≥ 50 agents with random pairwise interactions, the entropy of commitment types H(C) decreases over time, following approximately:

```
H(C, t) ≈ H(C, 0) · (1 - α·log(t))
```

for some convergence rate α > 0.

**Test:**
```python
def test_prediction_1(metrics: PopulationMetrics) -> Dict:
    """Test whether commitment entropy decreases logarithmically."""
    t = np.array(metrics.timesteps)
    H = np.array(metrics.commitment_entropy)
    
    # Fit: H = H0 * (1 - α*log(t))
    # Linearize: H/H0 = 1 - α*log(t)
    # So: 1 - H/H0 = α*log(t)
    
    H0 = H[0]
    y = 1 - H / H0
    x = np.log(t + 1)
    
    # Linear regression
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return {
        "prediction": "Protocol Convergence",
        "alpha": slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "passed": slope > 0 and p_value < 0.05 and r_value**2 > 0.5
    }
```

**Passing criterion:** α > 0, R² > 0.5, p < 0.05

---

### Prediction 2: Trust Network Structure

**Statement:** The emergent trust network will exhibit small-world properties:
- Clustering coefficient C > 0.3
- Average path length L < 2·log(N)

**Test:**
```python
def test_prediction_2(metrics: PopulationMetrics, n_agents: int) -> Dict:
    """Test whether trust network has small-world structure."""
    if not metrics.trust_network_metrics:
        return {"passed": False, "reason": "no_network_data"}
    
    final_network = metrics.trust_network_metrics[-1]
    
    clustering = final_network.get("clustering", 0)
    path_length = final_network.get("avg_path_length", float("inf"))
    
    threshold_path = 2 * np.log(n_agents)
    
    return {
        "prediction": "Small-World Trust Network",
        "clustering": clustering,
        "path_length": path_length,
        "path_length_threshold": threshold_path,
        "passed": clustering > 0.3 and path_length < threshold_path
    }
```

**Passing criterion:** C > 0.3 AND L < 2·log(N)

---

### Prediction 3: Template Fitness Dynamics

**Statement:** Templates with higher success rates will have higher adoption rates, following replicator dynamics:

```
∂p_i/∂t ∝ p_i · (f_i - f̄)
```

where p_i is template prevalence and f_i is fitness (success rate).

**Test:**
```python
def test_prediction_3(env: PopulationEnvironment) -> Dict:
    """Test whether template adoption follows replicator dynamics."""
    # Collect all templates and their stats
    templates = []
    for agent in env.agents:
        for t in agent.templates:
            templates.append({
                "id": t.id,
                "success_rate": t.success_rate,
                "times_used": t.times_used,
            })
    
    if len(templates) < 10:
        return {"passed": False, "reason": "insufficient_templates"}
    
    # Check correlation between success rate and usage
    success_rates = [t["success_rate"] for t in templates]
    usage = [t["times_used"] for t in templates]
    
    from scipy import stats
    correlation, p_value = stats.pearsonr(success_rates, usage)
    
    return {
        "prediction": "Template Replicator Dynamics",
        "correlation": correlation,
        "p_value": p_value,
        "passed": correlation > 0.3 and p_value < 0.05
    }
```

**Passing criterion:** Correlation > 0.3, p < 0.05

---

### Prediction 4: Specialization Emergence

**Statement:** Agent commitment portfolios will become less diverse over time. The Gini coefficient of commitment type distribution per agent will increase from ~0 to > 0.5 after T timesteps.

**Test:**
```python
def test_prediction_4(metrics: PopulationMetrics) -> Dict:
    """Test whether agents specialize over time."""
    if len(metrics.specialization_index) < 100:
        return {"passed": False, "reason": "insufficient_data"}
    
    early_spec = np.mean(metrics.specialization_index[:50])
    late_spec = np.mean(metrics.specialization_index[-50:])
    
    return {
        "prediction": "Specialization Emergence",
        "early_specialization": early_spec,
        "late_specialization": late_spec,
        "increase": late_spec - early_spec,
        "passed": late_spec > 0.5 and late_spec > early_spec + 0.2
    }
```

**Passing criterion:** Final specialization > 0.5 AND increase > 0.2

---

### Prediction 5: Critical Population Size

**Statement:** There exists N* ≈ 20-50 such that:
- For N < N*: No emergent structure (entropy stays high, no specialization)
- For N > N*: Emergent structure appears (entropy decreases, specialization increases)

**Test:**
```python
def test_prediction_5() -> Dict:
    """Test for critical population threshold."""
    population_sizes = [10, 20, 30, 50, 75, 100, 150, 200]
    results = []
    
    for n in population_sizes:
        config = PopulationConfig(n_agents=n, n_timesteps=5000)
        env = PopulationEnvironment(config)
        metrics = env.run()
        
        convergence = metrics.get_convergence_analysis()
        results.append({
            "n": n,
            "converged": convergence["converged"],
            "entropy_decrease": convergence["entropy_decrease"],
            "final_specialization": metrics.specialization_index[-1]
        })
    
    # Find transition point
    converged_at = [r["n"] for r in results if r["converged"]]
    not_converged_at = [r["n"] for r in results if not r["converged"]]
    
    if converged_at and not_converged_at:
        n_star = (max(not_converged_at) + min(converged_at)) / 2
    else:
        n_star = None
    
    return {
        "prediction": "Critical Population Size",
        "results": results,
        "n_star": n_star,
        "passed": n_star is not None and 15 < n_star < 75
    }
```

**Passing criterion:** Transition occurs between N=15 and N=75

---

## Part IV: The Institutional Emergence Narrative

### Why This Matters Beyond AI

If GCL reproduces institutional dynamics, it's not just an AI technique — it's a **computational model of coordination emergence**.

**Human parallels:**

| GCL Phenomenon | Human Institution |
|----------------|-------------------|
| Protocol convergence | Language/convention formation |
| Trust network formation | Social capital, reputation systems |
| Template propagation | Cultural evolution, meme spread |
| Agent specialization | Division of labor, expertise |
| Critical population size | Dunbar's number, market liquidity thresholds |

**The claim:** GCL captures fundamental principles that underlie coordination in both artificial and natural systems.

### Connection to Existing Literature

**Evolutionary game theory:**
- Replicator dynamics (Prediction 3) are the foundation of evolutionary game theory
- GCL templates are strategies; success rate is fitness
- Template propagation = strategy evolution

**Cultural evolution:**
- Boyd & Richerson: culture evolves via differential copying of successful variants
- GCL templates spread when they're successful
- This is computational cultural evolution

**Institutional economics:**
- North: institutions are "rules of the game"
- GCL commitments are explicit rules with enforcement (verification)
- Trust networks are emergent property rights

**Network science:**
- Small-world networks (Watts & Strogatz) enable efficient coordination
- GCL predicts small-world trust networks will emerge
- This connects to organizational theory

### Paper Framing

For maximum impact, the paper should:

1. **Present GCL as AI technique** (primary contribution)
2. **Show population-scale emergence** (novel validation)
3. **Connect to human institutions** (broad relevance)
4. **Discuss implications** (why this matters)

The emergence results transform the paper from "here's a coordination protocol" to "here's a computational theory of institutional formation."

---

## Part V: Experimental Protocol

### Full Experimental Suite

```python
# experiments/run_all.py

"""
Complete experimental suite for GCL paper.

Track 1: LLM Validation (API)
Track 2: Population Dynamics (Local)
"""

import asyncio
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("results")
RANDOM_SEEDS = [42, 123, 456, 789, 1011]  # 5 seeds for statistics

async def run_track_1_experiments():
    """
    Track 1: LLM experiments (requires API keys)
    
    Run these after local development is complete.
    Estimated cost: $500-1000
    """
    
    # Experiment A: Head-to-head comparison
    print("Running Experiment A: Head-to-head")
    await run_experiment_a_head_to_head(
        models=["claude-sonnet", "gpt-4", "claude-haiku"],
        conditions=["gcl", "nl_structured", "nl_free", "no_comm"],
        n_runs=100,
        seeds=RANDOM_SEEDS
    )
    
    # Experiment B: Drift robustness
    print("Running Experiment B: Drift robustness")
    await run_experiment_b_drift(
        model_pairs=[
            ("claude-sonnet", "claude-sonnet"),  # Same model
            ("claude-sonnet", "claude-haiku"),   # Same family
            ("claude-sonnet", "gpt-4"),          # Cross family
            ("gpt-4", "llama-70b"),              # Cross family
        ],
        n_runs=50,
        seeds=RANDOM_SEEDS
    )
    
    # Experiment D: Heterogeneous coordination
    print("Running Experiment D: Heterogeneous LLM coordination")
    await run_experiment_d_heterogeneous(
        agent_configs=[
            {"reasoning": "claude-sonnet", "retrieval": "gpt-4"},
            {"reasoning": "gpt-4", "retrieval": "claude-sonnet"},
        ],
        tasks=load_coordination_tasks(),
        n_runs=30,
        seeds=RANDOM_SEEDS
    )


def run_track_2_experiments():
    """
    Track 2: Population experiments (local compute)
    
    Run these first — no API costs.
    Estimated time: 2-4 hours on GPU, 8-12 hours on CPU
    """
    
    # Experiment H.1-H.4: Population dynamics
    print("Running Experiment H: Population dynamics")
    
    for n_agents in [50, 100, 200]:
        print(f"  N = {n_agents}")
        
        for seed in RANDOM_SEEDS:
            config = PopulationConfig(
                n_agents=n_agents,
                n_timesteps=10000,
                task_complexity=4
            )
            
            env = PopulationEnvironment(config)
            env.set_seed(seed)
            
            metrics = env.run()
            
            # Save results
            save_results(
                OUTPUT_DIR / f"population_n{n_agents}_seed{seed}.json",
                metrics
            )
    
    # Experiment H.5: Critical population threshold
    print("Running Experiment H.5: Critical threshold")
    run_experiment_h5_critical_threshold()
    
    # Test all predictions
    print("Testing predictions...")
    test_all_predictions()


def run_experiment_h5_critical_threshold():
    """Sweep population sizes to find critical threshold."""
    
    sizes = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
    results = []
    
    for n in sizes:
        print(f"  Testing N = {n}")
        
        seed_results = []
        for seed in RANDOM_SEEDS[:3]:  # Fewer seeds for sweep
            config = PopulationConfig(n_agents=n, n_timesteps=5000)
            env = PopulationEnvironment(config)
            env.set_seed(seed)
            metrics = env.run()
            
            seed_results.append({
                "converged": metrics.get_convergence_analysis()["converged"],
                "entropy_decrease": metrics.get_convergence_analysis()["entropy_decrease"],
                "final_specialization": metrics.specialization_index[-1]
            })
        
        results.append({
            "n": n,
            "convergence_rate": np.mean([r["converged"] for r in seed_results]),
            "avg_entropy_decrease": np.mean([r["entropy_decrease"] for r in seed_results]),
            "avg_specialization": np.mean([r["final_specialization"] for r in seed_results])
        })
    
    save_results(OUTPUT_DIR / "critical_threshold.json", results)


def test_all_predictions():
    """Run all prediction tests and generate report."""
    
    results = {}
    
    # Load most recent population run (N=100)
    metrics = load_results(OUTPUT_DIR / "population_n100_seed42.json")
    
    results["prediction_1"] = test_prediction_1(metrics)
    results["prediction_2"] = test_prediction_2(metrics, n_agents=100)
    results["prediction_4"] = test_prediction_4(metrics)
    
    # Prediction 3 needs environment
    config = PopulationConfig(n_agents=100, n_timesteps=10000)
    env = PopulationEnvironment(config)
    env.run()
    results["prediction_3"] = test_prediction_3(env)
    
    # Prediction 5 from sweep
    results["prediction_5"] = test_prediction_5()
    
    # Summary
    passed = sum(1 for r in results.values() if r.get("passed", False))
    total = len(results)
    
    print(f"\nPrediction Tests: {passed}/{total} passed")
    for name, result in results.items():
        status = "✓" if result.get("passed") else "✗"
        print(f"  {status} {result.get('prediction', name)}")
    
    save_results(OUTPUT_DIR / "prediction_tests.json", results)


if __name__ == "__main__":
    # Run local experiments first
    run_track_2_experiments()
    
    # Then run API experiments (uncomment when ready)
    # asyncio.run(run_track_1_experiments())
```

---

## Part VI: Updated Project Structure

```
gcl/
├── src/gcl/
│   ├── core/                    # Original core abstractions
│   │   ├── commitment.py
│   │   ├── verification.py
│   │   ├── reputation.py
│   │   └── template.py
│   ├── learning/                # Single-agent RL
│   │   ├── policy.py
│   │   ├── environment.py
│   │   └── training.py
│   ├── multiagent/              # Small-scale multi-agent
│   │   ├── market.py
│   │   └── matching.py
│   ├── population/              # NEW: Population-scale
│   │   ├── lightweight_agent.py
│   │   ├── environment.py
│   │   ├── metrics.py
│   │   └── analysis.py
│   ├── llm/                     # LLM integration
│   │   ├── adapter.py
│   │   └── caching.py
│   └── experiments/             # NEW: Organized experiments
│       ├── track1_llm/
│       │   ├── experiment_a_head_to_head.py
│       │   ├── experiment_b_drift.py
│       │   └── experiment_d_heterogeneous.py
│       ├── track2_population/
│       │   ├── experiment_h_dynamics.py
│       │   └── experiment_h5_threshold.py
│       └── predictions/
│           ├── test_predictions.py
│           └── statistical_analysis.py
├── results/                     # Experimental outputs
├── paper/
│   ├── main.tex
│   ├── figures/
│   │   ├── drift_robustness.py      # Generate money plot
│   │   ├── population_dynamics.py
│   │   └── trust_network.py
│   └── tables/
└── notebooks/                   # Analysis notebooks
    ├── 01_population_analysis.ipynb
    └── 02_prediction_validation.ipynb
```

---

## Part VII: Timeline Update

| Week | Focus | Track 1 (API) | Track 2 (Local) |
|------|-------|---------------|-----------------|
| 1-2 | Core implementation | — | Build + test |
| 3 | Templates + lightweight agents | — | Population env |
| 4 | Population experiments | — | **Run H.1-H.5** |
| 5 | LLM integration + baselines | **Run A** | Analysis |
| 6 | Drift experiments | **Run B** | Predictions |
| 7 | Full suite + writing | **Run D** | Figures |
| 8 | Polish + submit | Final runs | Paper draft |

**Key change:** Week 3-4 focuses on population experiments first (free, validates core claims), then Week 5-6 does LLM experiments (paid, validates practicality).

---

## Part VIII: Summary

This addendum provides:

1. **Lightweight agent architecture** — fast local experimentation
2. **Population environment** — 50-500 agent simulations
3. **Five falsifiable predictions** — specific, testable claims
4. **Statistical tests** — rigorous validation
5. **Institutional emergence narrative** — connects to human coordination
6. **Two-track experimental strategy** — local + API
7. **Complete experimental protocol** — reproducible science

**The paper now tells this story:**

> GCL is a framework for coordination via verifiable commitments. We prove it has better complexity than representation-based approaches (Theorem 1).
>
> At small scale, we show LLMs coordinate better with GCL than natural language (Track 1).
>
> At population scale, we observe emergent phenomena: protocols converge, trust networks form, templates propagate, agents specialize (Track 2).
>
> These dynamics mirror human institutions, suggesting GCL captures fundamental principles of coordination. We discuss implications for AI alignment.

**This is now a complete scientific contribution with theory, small-scale validation, and population-scale emergence.**

Continue building. The architecture supports everything we need.