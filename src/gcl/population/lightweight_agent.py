"""
Lightweight commitment agent for population-scale experiments.

This module provides fast, neural network-based agents that:
- Run fast (thousands of interactions per second)
- Learn commitment policies via simple RL
- Maintain reputation and templates
- Don't require API calls

These agents are NOT LLMs — they're small learned policies for
studying emergent coordination dynamics at scale (50-500 agents).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AgentConfig:
    """Configuration for lightweight commitment agent."""
    
    state_dim: int = 32          # Environment state dimension
    hidden_dim: int = 64         # Hidden layer size
    commitment_vocab: int = 16   # Number of commitment types
    template_capacity: int = 10  # Max templates to store
    learning_rate: float = 1e-3
    reputation_decay: float = 0.99  # Reputation decay per timestep
    max_stake_fraction: float = 0.5  # Max fraction of reputation to stake


class CommitmentEncoder(nn.Module):
    """
    Encode state into commitment logits.
    
    A small neural network that maps state observations to:
    - Commitment type probabilities
    - Confidence estimate
    - Stake amount
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        
        # Shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(config.state_dim + 4, config.hidden_dim),  # +4 for private state
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        # Output heads
        self.type_head = nn.Linear(config.hidden_dim, config.commitment_vocab)
        self.confidence_head = nn.Linear(config.hidden_dim, 1)
        self.stake_head = nn.Linear(config.hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            state: State tensor of shape (state_dim + 4,) or (batch, state_dim + 4)
            
        Returns:
            Dictionary with type_logits, confidence, and stake.
        """
        h = self.net(state)
        return {
            "type_logits": self.type_head(h),
            "confidence": torch.sigmoid(self.confidence_head(h)),
            "stake": nn.functional.softplus(self.stake_head(h)),
        }


@dataclass
class LightweightCommitment:
    """Simplified commitment for population experiments."""
    
    issuer: str
    commitment_type: int
    confidence: float
    stake: float
    template_id: Optional[str] = None
    state_hash: int = 0
    state_tensor: Optional[torch.Tensor] = None  # Set during creation


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
    """
    Template induced from successful commitments.
    
    KEY INSIGHT: Templates represent "commitment types that work" - they're
    about WHAT to commit to, not the specific state. This is learning-via-analogy:
    "This type of commitment has worked before, so use it again."
    
    The template tracks:
    - commitment_type: Which type of commitment this template represents
    - success_rate: How often this type succeeds (the "fitness")
    - times_used: How often this template has been used (for replicator dynamics)
    """
    
    id: str
    commitment_type: int
    prototype_state: torch.Tensor  # Kept for reference, but not used for matching
    success_rate: float
    context_variance: float
    times_used: int = 0
    times_succeeded: int = 0
    
    def match_score(self, state: torch.Tensor, commitment_type: int) -> float:
        """
        How well does this commitment match the template?
        
        SIMPLIFIED: Templates match by TYPE, not by state distance.
        If the commitment type matches, the template applies.
        
        Args:
            state: State tensor (not used in simplified matching)
            commitment_type: The commitment type being considered
            
        Returns:
            1.0 if type matches, 0.0 otherwise
        """
        # Type-based matching: templates are about commitment types, not states
        if commitment_type == self.commitment_type:
            return 1.0
        return 0.0
    
    def record_usage(self, success: bool) -> None:
        """Record template usage - CRITICAL for replicator dynamics."""
        self.times_used += 1
        if success:
            self.times_succeeded += 1
            # Update success rate with exponential moving average
            self.success_rate = 0.9 * self.success_rate + 0.1 * 1.0
        else:
            self.success_rate = 0.9 * self.success_rate + 0.1 * 0.0


@dataclass
class AgentStats:
    """Running statistics for analysis."""
    
    commitment_types_used: Dict[int, int] = field(default_factory=dict)
    success_by_type: Dict[int, List[bool]] = field(default_factory=dict)
    reputation_history: List[float] = field(default_factory=list)
    template_usage: Dict[str, int] = field(default_factory=dict)
    
    def update(self, commitment: LightweightCommitment, outcome: CommitmentOutcome) -> None:
        """Update statistics with new commitment outcome."""
        t = commitment.commitment_type
        self.commitment_types_used[t] = self.commitment_types_used.get(t, 0) + 1
        
        if t not in self.success_by_type:
            self.success_by_type[t] = []
        self.success_by_type[t].append(outcome.success)
        
        if commitment.template_id:
            self.template_usage[commitment.template_id] = \
                self.template_usage.get(commitment.template_id, 0) + 1
    
    def get_type_success_rate(self, commitment_type: int) -> float:
        """Get success rate for a commitment type."""
        if commitment_type not in self.success_by_type:
            return 0.5  # Prior
        successes = self.success_by_type[commitment_type]
        if not successes:
            return 0.5
        return sum(successes) / len(successes)


class LightweightCommitmentAgent:
    """
    Fast, lightweight agent for population experiments.
    
    Not an LLM — a small learned policy that:
    - Proposes commitments based on state
    - Tracks its own reputation
    - Maintains a library of learned templates
    - Updates policy via simple RL (REINFORCE)
    
    Example:
        >>> config = AgentConfig(state_dim=32, commitment_vocab=16)
        >>> agent = LightweightCommitmentAgent("agent_0", config)
        >>> state = torch.randn(36)  # 32 + 4 private state
        >>> commitment = agent.propose_commitment(state, partner_reputation=1.0)
        >>> agent.receive_outcome(commitment, outcome, reward)
    """
    
    def __init__(self, agent_id: str, config: AgentConfig | None = None):
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique identifier for this agent.
            config: Agent configuration.
        """
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
        partner_reputation: float = 1.0,
    ) -> LightweightCommitment:
        """
        Propose a commitment given current state.
        
        The agent considers:
        - Current state (what needs to be done)
        - Own reputation (how much can I stake?)
        - Partner reputation (how much should I trust them?)
        - Available templates (what patterns have worked?)
        
        REPLICATOR DYNAMICS: Templates with higher success rates get used more.
        This is the key mechanism for template fitness selection.
        
        Args:
            state: State tensor (should include private state).
            partner_reputation: Reputation of interaction partner.
            
        Returns:
            A lightweight commitment.
        """
        with torch.no_grad():
            outputs = self.encoder(state)
        
        # Get base type probabilities from policy
        type_probs = torch.softmax(outputs["type_logits"], dim=-1).flatten()
        
        # REPLICATOR DYNAMICS: Strongly boost probabilities for successful templates
        # Templates with higher success rates should be used proportionally more
        # This implements: ∂p_i/∂t ∝ p_i · (f_i - f̄)
        if self.templates:
            template_boost = torch.zeros(self.config.commitment_vocab)
            
            # Calculate mean fitness (success rate) across templates
            if len(self.templates) > 0:
                mean_fitness = np.mean([t.success_rate for t in self.templates])
            else:
                mean_fitness = 0.5
            
            for template in self.templates:
                # Replicator dynamics: boost proportional to (fitness - mean_fitness)
                # Templates above average get boosted, below average get reduced
                fitness_advantage = template.success_rate - mean_fitness
                
                # Strong boost for high-success templates (key for replicator dynamics)
                # Use exponential to create strong selection pressure
                if template.success_rate > 0.3:  # Only boost reasonably successful templates
                    boost = np.exp(2.0 * fitness_advantage) - 1.0  # Exponential selection
                    boost = max(0, boost)  # No negative boosts
                    template_boost[template.commitment_type] += boost
            
            # Apply boost (multiplicative) with stronger effect
            type_probs = type_probs * (1.0 + template_boost)
            type_probs = type_probs / type_probs.sum()  # Renormalize
        
        commitment_type = int(torch.multinomial(type_probs, 1).item())
        
        # Determine confidence and stake
        confidence = float(outputs["confidence"].item())
        base_stake = float(outputs["stake"].item())
        
        # Modulate stake by reputation (can't stake more than you have)
        max_stake = self.reputation * self.config.max_stake_fraction
        actual_stake = min(base_stake, max_stake)
        
        # Check if we have a relevant template for the chosen type
        template = self._find_matching_template(state, commitment_type)
        
        commitment = LightweightCommitment(
            issuer=self.agent_id,
            commitment_type=commitment_type,
            confidence=confidence,
            stake=actual_stake,
            template_id=template.id if template else None,
            state_hash=self._hash_state(state),
            state_tensor=state.clone(),
        )
        
        return commitment
    
    def receive_outcome(
        self,
        commitment: LightweightCommitment,
        outcome: CommitmentOutcome,
        reward: float,
    ) -> None:
        """
        Update agent based on commitment outcome.
        
        Args:
            commitment: The commitment that was made.
            outcome: The outcome of the commitment.
            reward: Reward received.
        """
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
            reward=reward,
        ))
        
        # Update stats
        self.stats.update(commitment, outcome)
        self.stats.reputation_history.append(self.reputation)
        
        # Update template if used
        if commitment.template_id:
            template = next(
                (t for t in self.templates if t.id == commitment.template_id),
                None
            )
            if template:
                template.record_usage(outcome.success)
        
        # Induce templates more frequently - every 10 commitments
        # This ensures templates form early and get used
        if len(self.commitment_history) % 10 == 0:
            self._maybe_induce_template()
    
    def update_policy(self, batch: List[CommitmentRecord] | None = None) -> float:
        """
        Policy gradient update from batch of experiences.
        
        Simple REINFORCE with baseline.
        
        Args:
            batch: Batch of commitment records. If None, uses recent history.
            
        Returns:
            Loss value.
        """
        if batch is None:
            batch = self.commitment_history[-20:] if len(self.commitment_history) >= 20 else []
        
        if len(batch) < 10:
            return 0.0
        
        self.optimizer.zero_grad()
        
        total_loss = torch.tensor(0.0)
        baseline = np.mean([r.reward for r in batch])
        
        for record in batch:
            if record.commitment.state_tensor is None:
                continue
                
            state = record.commitment.state_tensor
            outputs = self.encoder(state)
            
            # Log probability of taken action
            type_logits = outputs["type_logits"]
            log_prob = torch.log_softmax(type_logits.flatten(), dim=-1)[record.commitment.commitment_type]
            
            # Policy gradient
            advantage = record.reward - baseline
            loss = -log_prob * advantage
            total_loss = total_loss + loss
        
        if len(batch) > 0:
            total_loss = total_loss / len(batch)
            total_loss.backward()
            self.optimizer.step()
        
        return float(total_loss.item())
    
    def _find_matching_template(
        self,
        state: torch.Tensor,
        commitment_type: int,
    ) -> Optional[LearnedTemplate]:
        """
        Find a template that matches the commitment type.
        
        SIMPLIFIED: Templates match by TYPE, not by state.
        If we have a template for this commitment type, use it.
        """
        for template in self.templates:
            if template.commitment_type == commitment_type:
                return template
        return None
    
    def _maybe_induce_template(self) -> Optional[LearnedTemplate]:
        """
        Induce a new template from recent successful commitments.
        
        AGGRESSIVE TEMPLATE FORMATION: We want templates to form early
        so they can be used and tracked for replicator dynamics.
        
        Templates represent "commitment types that work" - the key insight
        from learning-via-analogy.
        
        Returns:
            Newly induced template, or None.
        """
        # Use shorter history window for faster template formation
        recent = self.commitment_history[-50:]
        successes = [r for r in recent if r.outcome.success]
        
        # Very low threshold - we want templates to form early
        if len(successes) < 2:
            return None
        
        # Cluster by commitment type
        by_type: Dict[int, List[CommitmentRecord]] = {}
        for s in successes:
            t = s.commitment.commitment_type
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(s)
        
        # For each type with ANY successes, create template
        for commit_type, records in by_type.items():
            # Just need 2 successes to form a template
            if len(records) < 2:
                continue
            
            # Calculate success rate for this type
            type_total = len([r for r in recent if r.commitment.commitment_type == commit_type])
            success_rate = len(records) / max(1, type_total)
            
            # Low threshold - even 30% success rate is worth templating
            if success_rate < 0.3:
                continue
            
            # Get states for prototype (just need 1)
            states = [r.commitment.state_tensor for r in records if r.commitment.state_tensor is not None]
            if len(states) < 1:
                continue
                
            state_stack = torch.stack(states)
            state_var = torch.var(state_stack, dim=0).mean().item() if len(states) > 1 else 1.0
            
            # ALWAYS create template if we have successes - no variance check
            # The key is the commitment TYPE, not the state pattern
            template = LearnedTemplate(
                id=f"{self.agent_id}_t{len(self.templates)}_{commit_type}",
                commitment_type=commit_type,
                prototype_state=torch.mean(state_stack, dim=0),
                success_rate=success_rate,
                context_variance=max(1.0, state_var),
            )
            
            # Check if we already have a template for this type
            existing = next(
                (t for t in self.templates if t.commitment_type == commit_type),
                None
            )
            if existing:
                # Update existing if new is better
                if success_rate > existing.success_rate:
                    self.templates.remove(existing)
                    self.templates.append(template)
                    return template
            elif len(self.templates) < self.config.template_capacity:
                self.templates.append(template)
                return template
            else:
                worst = min(self.templates, key=lambda t: t.success_rate)
                if success_rate > worst.success_rate:
                    self.templates.remove(worst)
                    self.templates.append(template)
                    return template
        
        return None
    
    def _hash_state(self, state: torch.Tensor) -> int:
        """Simple state hash for commitment identification."""
        return hash(tuple(state.detach().numpy().round(2).flatten().tolist()))
    
    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        if self.total_commitments == 0:
            return 0.5
        return self.fulfilled_commitments / self.total_commitments
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get agent state for serialization."""
        return {
            "agent_id": self.agent_id,
            "config": self.config,
            "encoder_state": self.encoder.state_dict(),
            "reputation": self.reputation,
            "total_commitments": self.total_commitments,
            "fulfilled_commitments": self.fulfilled_commitments,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load agent state from serialization."""
        self.encoder.load_state_dict(state_dict["encoder_state"])
        self.reputation = state_dict["reputation"]
        self.total_commitments = state_dict["total_commitments"]
        self.fulfilled_commitments = state_dict["fulfilled_commitments"]
