"""
Experiment 33: Specialization and Template Dynamics

Research Question: What drives autonomous task selection?
- Are agents specializing in certain task types?
- Is specialization at the content level (task type) or template level (strategy)?
- How does template complexity affect agent-template interactions?

Design:
1. Introduce task types (technical, social, creative)
2. Track agent specialization over time
3. Vary template complexity (simple → complex over time)
4. Measure agent-template affinity and specialization emergence

Factors:
- Task type diversity: 1 type (homogeneous) vs 3 types (heterogeneous)
- Template complexity: fixed vs increasing over time
- Specialization pressure: none vs rewarded

Metrics:
- Specialization index (Herfindahl-Hirschman Index on task type distribution)
- Template-task affinity (correlation between template type and task type)
- Capability-specialization correlation
- Cooperation by specialization level
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

from experiments.social_structures.agents.agent import Agent, Task, TaskOutcome, Template, create_population
from experiments.social_structures.structures.base import BaseStructure
from experiments.social_structures.competition.firm import Firm


class TaskType(Enum):
    """Types of tasks agents can specialize in."""
    TECHNICAL = "technical"
    SOCIAL = "social"
    CREATIVE = "creative"


class TemplateComplexity(Enum):
    """How template complexity evolves."""
    FIXED_SIMPLE = "fixed_simple"      # All templates simple (0.05-0.10 boost)
    FIXED_COMPLEX = "fixed_complex"    # All templates complex (0.10-0.20 boost)
    INCREASING = "increasing"          # Complexity increases over time
    MIXED = "mixed"                    # Mix of simple and complex


@dataclass
class SpecializedTemplate:
    """Template with task type affinity and complexity."""
    id: str
    capability_boost: float
    task_type: TaskType
    complexity: float  # 0-1, affects learning difficulty and power
    source_agent: Optional[str] = None
    generation: int = 0  # When it was created (for tracking evolution)
    
    def __eq__(self, other):
        if not isinstance(other, SpecializedTemplate):
            return False
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class SpecializedTask:
    """Task with type and complexity requirements."""
    id: str
    difficulty: float
    reward: float
    task_type: TaskType
    complexity_requirement: float  # Minimum template complexity needed for bonus
    
    def to_base_task(self) -> Task:
        """Convert to base Task for compatibility."""
        return Task(
            id=self.id,
            difficulty=self.difficulty,
            reward=self.reward,
            task_type=self.task_type.value
        )


@dataclass
class SpecializationConfig:
    """Configuration for specialization experiment."""
    task_diversity: int = 3  # Number of task types (1 or 3)
    template_complexity: TemplateComplexity = TemplateComplexity.INCREASING
    specialization_bonus: float = 0.2  # Bonus for matching template to task type
    complexity_bonus: float = 0.1  # Bonus for complex templates on complex tasks
    success_reward: float = 0.1
    failure_penalty: float = 0.3


@dataclass
class AgentSpecializationState:
    """Track an agent's specialization over time."""
    agent_id: str
    task_type_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    template_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    success_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failure_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def record_attempt(self, task_type: TaskType, success: bool):
        """Record a task attempt."""
        self.task_type_counts[task_type.value] += 1
        if success:
            self.success_by_type[task_type.value] += 1
        else:
            self.failure_by_type[task_type.value] += 1
    
    def add_template(self, template: SpecializedTemplate):
        """Record a new template."""
        self.template_types[template.task_type.value] += 1
    
    def get_specialization_index(self) -> float:
        """
        Calculate Herfindahl-Hirschman Index for task type distribution.
        HHI = sum(share^2) where share is fraction of attempts in each type.
        HHI = 1.0 means perfect specialization (all one type)
        HHI = 0.33 means even distribution across 3 types
        """
        total = sum(self.task_type_counts.values())
        if total == 0:
            return 0.0
        shares = [count / total for count in self.task_type_counts.values()]
        return sum(s ** 2 for s in shares)
    
    def get_dominant_type(self) -> Optional[TaskType]:
        """Get the task type this agent does most."""
        if not self.task_type_counts:
            return None
        dominant = max(self.task_type_counts.items(), key=lambda x: x[1])
        return TaskType(dominant[0])
    
    def get_success_rate_by_type(self) -> Dict[str, float]:
        """Get success rate for each task type."""
        rates = {}
        for task_type in TaskType:
            successes = self.success_by_type[task_type.value]
            failures = self.failure_by_type[task_type.value]
            total = successes + failures
            rates[task_type.value] = successes / total if total > 0 else 0.0
        return rates


class SpecializationStructure(BaseStructure):
    """
    Structure that tracks specialization dynamics.
    """
    
    def __init__(self, config: SpecializationConfig):
        self.config = config
        self._name = f"spec_{config.task_diversity}types_{config.template_complexity.value}"
        
        # Track specialization state for each agent
        self.agent_states: Dict[str, AgentSpecializationState] = {}
        
        # Track template evolution
        self.template_generations: List[int] = []
        self.current_round = 0
        
        # Metrics over time
        self.specialization_over_time: List[float] = []
        self.cooperation_by_specialization: Dict[str, List[float]] = {
            "low": [], "medium": [], "high": []
        }
        
    @property
    def name(self) -> str:
        return self._name
    
    def get_agent_state(self, agent: Agent) -> AgentSpecializationState:
        """Get or create specialization state for an agent."""
        if agent.id not in self.agent_states:
            self.agent_states[agent.id] = AgentSpecializationState(agent_id=agent.id)
        return self.agent_states[agent.id]
    
    def generate_task(self, round_num: int) -> SpecializedTask:
        """Generate a task with type and complexity."""
        self.current_round = round_num
        
        # Select task type
        if self.config.task_diversity == 1:
            task_type = TaskType.TECHNICAL
        else:
            task_type = random.choice(list(TaskType))
        
        # Determine complexity requirement based on round
        if self.config.template_complexity == TemplateComplexity.FIXED_SIMPLE:
            complexity_req = 0.3
        elif self.config.template_complexity == TemplateComplexity.FIXED_COMPLEX:
            complexity_req = 0.7
        elif self.config.template_complexity == TemplateComplexity.INCREASING:
            # Complexity increases over time
            complexity_req = 0.3 + 0.5 * (round_num / 100)
            complexity_req = min(0.8, complexity_req)
        else:  # MIXED
            complexity_req = random.uniform(0.3, 0.8)
        
        difficulty = random.uniform(0.3, 0.7)
        
        return SpecializedTask(
            id=f"task_{round_num}",
            difficulty=difficulty,
            reward=1.0,
            task_type=task_type,
            complexity_requirement=complexity_req
        )
    
    def get_template_boost(self, agent: Agent, task: SpecializedTask) -> float:
        """
        Calculate template boost considering specialization.
        
        Boost = base_boost + specialization_bonus (if template matches task type)
                          + complexity_bonus (if template complexity >= requirement)
        """
        total_boost = 0.0
        
        for template in agent.template_library:
            if isinstance(template, SpecializedTemplate):
                base = template.capability_boost
                
                # Specialization bonus
                if template.task_type == task.task_type:
                    base += self.config.specialization_bonus
                
                # Complexity bonus
                if template.complexity >= task.complexity_requirement:
                    base += self.config.complexity_bonus
                
                total_boost += base
            else:
                # Regular template
                total_boost += template.capability_boost
        
        return total_boost
    
    def calculate_success_probability(
        self, 
        agent: Agent, 
        task: SpecializedTask,
        effort: float = 0.8
    ) -> float:
        """Calculate success probability with specialization effects."""
        base_cap = agent.base_capability + agent.learned_capability
        template_boost = self.get_template_boost(agent, task)
        effective_cap = min(1.0, base_cap + template_boost)
        
        success_prob = effective_cap * effort * (1 - task.difficulty * 0.6)
        return max(0.0, min(1.0, success_prob))
    
    def get_volunteers(self, task: Task, agents: List[Agent]) -> List[Agent]:
        """
        Agents volunteer based on specialization match.
        
        Self-selection considers:
        1. Own capability
        2. Template match to task type
        3. Past success with this task type
        """
        available = self.get_available_agents(agents)
        if not available:
            return []
        
        # Convert to specialized task if needed
        if isinstance(task, SpecializedTask):
            spec_task = task
        else:
            spec_task = SpecializedTask(
                id=task.id,
                difficulty=task.difficulty,
                reward=task.reward,
                task_type=TaskType(task.task_type) if task.task_type in [t.value for t in TaskType] else TaskType.TECHNICAL,
                complexity_requirement=0.5
            )
        
        volunteers = []
        for agent in available:
            state = self.get_agent_state(agent)
            
            # Calculate volunteer score
            base_cap = agent.effective_capability
            
            # Bonus for matching templates
            template_match = 0.0
            for template in agent.template_library:
                if isinstance(template, SpecializedTemplate):
                    if template.task_type == spec_task.task_type:
                        template_match += 0.1
            
            # Bonus for past success with this type
            success_rates = state.get_success_rate_by_type()
            past_success = success_rates.get(spec_task.task_type.value, 0.5)
            
            # Combined score
            volunteer_score = base_cap + template_match + past_success * 0.2
            
            # Volunteer if score exceeds threshold
            if volunteer_score > spec_task.difficulty * 0.6:
                volunteers.append((agent, volunteer_score))
        
        if not volunteers:
            # No volunteers - pick highest capability
            return [max(available, key=lambda a: a.effective_capability)]
        
        # Pick best volunteer
        best = max(volunteers, key=lambda x: x[1])
        return [best[0]]
    
    def attempt_specialized_task(
        self, 
        agent: Agent, 
        task: SpecializedTask
    ) -> TaskOutcome:
        """Attempt a specialized task."""
        state = self.get_agent_state(agent)
        
        # Calculate success probability
        success_prob = self.calculate_success_probability(agent, task)
        success = random.random() < success_prob
        
        # Record attempt
        state.record_attempt(task.task_type, success)
        
        # Update agent history
        if success:
            agent.success_history.append(True)
            output_quality = agent.effective_capability * (1 + random.uniform(-0.1, 0.1))
        else:
            agent.failure_history.append(True)
            output_quality = 0.0
        
        return TaskOutcome(
            success=success,
            agent=agent,
            task=task.to_base_task(),
            output_quality=output_quality
        )
    
    def learn_specialized_template(
        self, 
        agent: Agent, 
        task: SpecializedTask, 
        success: bool
    ) -> Optional[SpecializedTemplate]:
        """Generate a specialized template from experience."""
        if not success:
            return None
        
        if random.random() > 0.1:  # 10% chance
            return None
        
        # Determine template complexity based on config
        if self.config.template_complexity == TemplateComplexity.FIXED_SIMPLE:
            complexity = random.uniform(0.2, 0.4)
            boost = random.uniform(0.05, 0.10)
        elif self.config.template_complexity == TemplateComplexity.FIXED_COMPLEX:
            complexity = random.uniform(0.6, 0.9)
            boost = random.uniform(0.10, 0.20)
        elif self.config.template_complexity == TemplateComplexity.INCREASING:
            # Complexity increases with round
            base_complexity = 0.3 + 0.5 * (self.current_round / 100)
            complexity = min(0.9, base_complexity + random.uniform(-0.1, 0.1))
            boost = 0.05 + 0.15 * complexity
        else:  # MIXED
            complexity = random.uniform(0.2, 0.9)
            boost = 0.05 + 0.15 * complexity
        
        template = SpecializedTemplate(
            id=f"spec_template_{agent.id}_{len(agent.template_library)}_{random.randint(0, 10000)}",
            capability_boost=boost,
            task_type=task.task_type,
            complexity=complexity,
            source_agent=agent.id,
            generation=self.current_round
        )
        
        # Add to agent's library
        agent.template_library.append(template)
        
        # Track in state
        state = self.get_agent_state(agent)
        state.add_template(template)
        
        self.template_generations.append(self.current_round)
        
        return template
    
    def process_outcome_with_difficulty(
        self,
        agent: Agent,
        outcome: TaskOutcome,
        all_agents: List[Agent],
        difficulty: float
    ):
        """Update reputation based on outcome."""
        difficulty_mult = 1.0 + difficulty * self.difficulty_reputation_multiplier
        
        if outcome.success:
            delta = self.config.success_reward * difficulty_mult
            agent.reputation = min(1.0, agent.reputation + delta)
        else:
            delta = self.config.failure_penalty / difficulty_mult
            agent.reputation = max(0.0, agent.reputation - delta)
    
    def maybe_share_knowledge(self, agents: List[Agent]):
        """Share templates (full pooling for this experiment)."""
        all_template_ids = set()
        template_map = {}
        
        for agent in agents:
            for t in agent.template_library:
                if t.id not in all_template_ids:
                    all_template_ids.add(t.id)
                    template_map[t.id] = t
        
        for agent in agents:
            agent_ids = {t.id for t in agent.template_library}
            for tid, template in template_map.items():
                if tid not in agent_ids:
                    agent.template_library.append(template)
    
    def aggregate_outputs(self, outcomes: List[TaskOutcome]) -> float:
        return sum(o.output_quality for o in outcomes if o.success)
    
    def transfer_knowledge(self, from_agent: Agent, to_agent: Agent) -> bool:
        if from_agent.template_library:
            template = random.choice(from_agent.template_library)
            if template not in to_agent.template_library:
                to_agent.template_library.append(template)
                return True
        return False
    
    def calculate_status(self, agent: Agent, all_agents: List[Agent]) -> str:
        return self.status_from_percentile(agent, all_agents)
    
    def check_obligations(self, agent: Agent, all_agents: List[Agent]):
        pass
    
    def get_population_specialization(self) -> float:
        """Get average specialization index across all agents."""
        if not self.agent_states:
            return 0.0
        indices = [state.get_specialization_index() for state in self.agent_states.values()]
        return sum(indices) / len(indices)
    
    def get_template_type_distribution(self) -> Dict[str, int]:
        """Get distribution of template types across all agents."""
        dist = defaultdict(int)
        for state in self.agent_states.values():
            for task_type, count in state.template_types.items():
                dist[task_type] += count
        return dict(dist)


def run_condition(
    config: SpecializationConfig,
    n_rounds: int = 100,
    n_agents: int = 30,
    seed: int = 0
) -> Dict[str, Any]:
    """Run a single experimental condition."""
    random.seed(seed)
    np.random.seed(seed)
    
    structure = SpecializationStructure(config)
    agents = create_population(n_agents, prefix=f"{structure.name}_{seed}")
    
    # Track metrics over time
    specialization_over_time = []
    cooperation_over_time = []
    template_complexity_over_time = []
    
    for round_num in range(n_rounds):
        # Generate specialized task
        task = structure.generate_task(round_num)
        
        # Get volunteers
        volunteers = structure.get_volunteers(task.to_base_task(), agents)
        
        if volunteers:
            agent = volunteers[0]
            
            # Attempt task
            outcome = structure.attempt_specialized_task(agent, task)
            
            # Update reputation
            structure.process_outcome_with_difficulty(
                agent, outcome, agents, task.difficulty
            )
            
            # Learn template
            structure.learn_specialized_template(agent, task, outcome.success)
        
        # Share knowledge
        structure.maybe_share_knowledge(agents)
        
        # Track metrics every 10 rounds
        if round_num % 10 == 0:
            spec_idx = structure.get_population_specialization()
            specialization_over_time.append(spec_idx)
            
            # Calculate cooperation rate
            total = sum(len(a.success_history) + len(a.failure_history) for a in agents)
            successes = sum(len(a.success_history) for a in agents)
            coop = successes / total if total > 0 else 0
            cooperation_over_time.append(coop)
            
            # Track template complexity
            complexities = []
            for agent in agents:
                for t in agent.template_library:
                    if isinstance(t, SpecializedTemplate):
                        complexities.append(t.complexity)
            avg_complexity = sum(complexities) / len(complexities) if complexities else 0
            template_complexity_over_time.append(avg_complexity)
    
    # Final metrics
    n = len(agents)
    total_tasks = sum(len(a.success_history) + len(a.failure_history) for a in agents)
    successes = sum(len(a.success_history) for a in agents)
    cooperation_rate = successes / total_tasks if total_tasks > 0 else 0
    
    mean_capability = sum(a.effective_capability for a in agents) / n if n > 0 else 0
    
    # Specialization metrics
    final_specialization = structure.get_population_specialization()
    template_dist = structure.get_template_type_distribution()
    
    # Agent-level specialization analysis
    agent_specializations = []
    agent_success_rates = []
    for agent in agents:
        state = structure.get_agent_state(agent)
        spec_idx = state.get_specialization_index()
        agent_specializations.append(spec_idx)
        
        # Success rate for dominant type
        dominant = state.get_dominant_type()
        if dominant:
            rates = state.get_success_rate_by_type()
            agent_success_rates.append(rates.get(dominant.value, 0))
    
    # Correlation between specialization and success
    if len(agent_specializations) > 1 and len(agent_success_rates) > 1:
        spec_success_corr = np.corrcoef(agent_specializations, agent_success_rates)[0, 1]
    else:
        spec_success_corr = 0.0
    
    return {
        "cooperation_rate": cooperation_rate,
        "mean_capability": mean_capability,
        "final_specialization": final_specialization,
        "specialization_over_time": specialization_over_time,
        "cooperation_over_time": cooperation_over_time,
        "template_complexity_over_time": template_complexity_over_time,
        "template_distribution": template_dist,
        "spec_success_correlation": spec_success_corr if not np.isnan(spec_success_corr) else 0.0,
        "n_specialized_templates": sum(1 for a in agents for t in a.template_library if isinstance(t, SpecializedTemplate)),
    }


def run_experiment(
    n_rounds: int = 100,
    n_seeds: int = 10,
    n_agents: int = 30
) -> Dict[str, Any]:
    """Run the full specialization experiment."""
    print("=" * 70)
    print("EXPERIMENT 33: SPECIALIZATION AND TEMPLATE DYNAMICS")
    print("=" * 70)
    print(f"Rounds: {n_rounds}, Seeds: {n_seeds}, Agents: {n_agents}")
    print()
    
    results = {}
    
    # Test conditions
    conditions = [
        # Baseline: 1 type, fixed simple
        SpecializationConfig(task_diversity=1, template_complexity=TemplateComplexity.FIXED_SIMPLE),
        # 3 types, fixed simple
        SpecializationConfig(task_diversity=3, template_complexity=TemplateComplexity.FIXED_SIMPLE),
        # 3 types, fixed complex
        SpecializationConfig(task_diversity=3, template_complexity=TemplateComplexity.FIXED_COMPLEX),
        # 3 types, increasing complexity
        SpecializationConfig(task_diversity=3, template_complexity=TemplateComplexity.INCREASING),
        # 3 types, mixed complexity
        SpecializationConfig(task_diversity=3, template_complexity=TemplateComplexity.MIXED),
    ]
    
    for config in conditions:
        name = f"{config.task_diversity}types_{config.template_complexity.value}"
        print(f"Running {name}...", end=" ", flush=True)
        
        seed_results = []
        for seed in range(n_seeds):
            metrics = run_condition(config, n_rounds, n_agents, seed)
            seed_results.append(metrics)
        
        # Aggregate scalar metrics
        aggregated = {}
        scalar_keys = ["cooperation_rate", "mean_capability", "final_specialization", 
                       "spec_success_correlation", "n_specialized_templates"]
        for key in scalar_keys:
            values = [r[key] for r in seed_results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        
        # Aggregate time series (average across seeds)
        for key in ["specialization_over_time", "cooperation_over_time", "template_complexity_over_time"]:
            arrays = [np.array(r[key]) for r in seed_results]
            min_len = min(len(a) for a in arrays)
            trimmed = [a[:min_len] for a in arrays]
            aggregated[key] = {
                "mean": np.mean(trimmed, axis=0).tolist(),
                "std": np.std(trimmed, axis=0).tolist(),
            }
        
        results[name] = aggregated
        print(f"coop={aggregated['cooperation_rate']['mean']:.3f}, "
              f"spec={aggregated['final_specialization']['mean']:.3f}, "
              f"corr={aggregated['spec_success_correlation']['mean']:.3f}")
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze specialization dynamics."""
    analysis = {}
    
    # Compare specialization levels
    spec_levels = {name: r["final_specialization"]["mean"] for name, r in results.items()}
    analysis["specialization_by_condition"] = spec_levels
    
    # Compare cooperation rates
    coop_rates = {name: r["cooperation_rate"]["mean"] for name, r in results.items()}
    analysis["cooperation_by_condition"] = coop_rates
    
    # Correlation between specialization and cooperation across conditions
    specs = list(spec_levels.values())
    coops = list(coop_rates.values())
    if len(specs) > 1:
        analysis["spec_coop_correlation"] = float(np.corrcoef(specs, coops)[0, 1])
    else:
        analysis["spec_coop_correlation"] = 0.0
    
    # Effect of task diversity
    single_type = [r for name, r in results.items() if "1types" in name]
    multi_type = [r for name, r in results.items() if "3types" in name]
    
    if single_type and multi_type:
        single_spec = np.mean([r["final_specialization"]["mean"] for r in single_type])
        multi_spec = np.mean([r["final_specialization"]["mean"] for r in multi_type])
        analysis["diversity_effect_on_specialization"] = multi_spec - single_spec
    
    # Effect of complexity
    simple = [r for name, r in results.items() if "fixed_simple" in name and "3types" in name]
    complex_ = [r for name, r in results.items() if "fixed_complex" in name]
    increasing = [r for name, r in results.items() if "increasing" in name]
    
    if simple and complex_:
        simple_coop = np.mean([r["cooperation_rate"]["mean"] for r in simple])
        complex_coop = np.mean([r["cooperation_rate"]["mean"] for r in complex_])
        analysis["complexity_effect_on_cooperation"] = complex_coop - simple_coop
    
    if increasing:
        inc_coop = np.mean([r["cooperation_rate"]["mean"] for r in increasing])
        analysis["increasing_complexity_cooperation"] = inc_coop
    
    return analysis


def print_results(results: Dict[str, Any], analysis: Dict[str, Any]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Specialization and Template Dynamics")
    print("=" * 70)
    
    # Main results table
    print("\n--- Results by Condition ---")
    print(f"{'Condition':>30} {'Cooperation':>12} {'Specialization':>14} {'Spec-Success':>12}")
    print("-" * 72)
    
    for name, metrics in results.items():
        coop = metrics["cooperation_rate"]["mean"]
        spec = metrics["final_specialization"]["mean"]
        corr = metrics["spec_success_correlation"]["mean"]
        print(f"{name:>30} {coop:>12.3f} {spec:>14.3f} {corr:>12.3f}")
    
    # Analysis
    print("\n--- Analysis ---")
    
    print(f"\nSpecialization-Cooperation Correlation: {analysis['spec_coop_correlation']:.3f}")
    
    if "diversity_effect_on_specialization" in analysis:
        print(f"Task Diversity Effect on Specialization: {analysis['diversity_effect_on_specialization']:+.3f}")
    
    if "complexity_effect_on_cooperation" in analysis:
        print(f"Complexity Effect on Cooperation: {analysis['complexity_effect_on_cooperation']:+.3f}")
    
    # Key findings
    print("\n--- Key Findings ---")
    
    # Does specialization emerge?
    multi_type_specs = [r["final_specialization"]["mean"] for name, r in results.items() if "3types" in name]
    if multi_type_specs:
        avg_spec = np.mean(multi_type_specs)
        if avg_spec > 0.5:
            print(f"✓ Specialization EMERGES (avg HHI = {avg_spec:.3f} > 0.5)")
        else:
            print(f"✗ Specialization does NOT emerge strongly (avg HHI = {avg_spec:.3f})")
    
    # Does specialization help?
    if analysis["spec_coop_correlation"] > 0.3:
        print(f"✓ Specialization HELPS cooperation (r = {analysis['spec_coop_correlation']:.3f})")
    elif analysis["spec_coop_correlation"] < -0.3:
        print(f"✗ Specialization HURTS cooperation (r = {analysis['spec_coop_correlation']:.3f})")
    else:
        print(f"○ Specialization has WEAK effect on cooperation (r = {analysis['spec_coop_correlation']:.3f})")
    
    # Does complexity matter?
    if "complexity_effect_on_cooperation" in analysis:
        effect = analysis["complexity_effect_on_cooperation"]
        if effect > 0.05:
            print(f"✓ Complex templates HELP cooperation ({effect:+.3f})")
        elif effect < -0.05:
            print(f"✗ Complex templates HURT cooperation ({effect:+.3f})")
        else:
            print(f"○ Template complexity has MINIMAL effect ({effect:+.3f})")


def main():
    """Run the full experiment."""
    results = run_experiment(
        n_rounds=100,
        n_seeds=10,
        n_agents=30
    )
    
    # Analyze
    analysis = analyze_results(results)
    
    # Print results
    print_results(results, analysis)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    output = {
        "conditions": results,
        "analysis": analysis,
    }
    
    with open("results/experiment_33_specialization.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    print("\nResults saved to results/experiment_33_specialization.json")
    
    return output


if __name__ == "__main__":
    main()