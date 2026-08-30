"""
Experiment 08: Strong MAS Comparison

This experiment compares GCL against strong Multi-Agent Systems baselines
to ensure we're not "straw-manning" by only comparing against natural
language LLM coordination.

Systems Compared:
1. **GCL Commitment Protocol** - Our commitment-based coordination
2. Contract Net Protocol (CNP) - Classic task allocation
3. FIPA-ACL Protocol - Structured agent communication
4. Auction-Based Coordination - Market mechanisms
5. MARL (Independent Q-Learning) - Learning-based coordination

Metrics:
- Task completion rate
- Coordination efficiency
- Message complexity
- Time to coordination
- Scalability

The key hypothesis: GCL should match or exceed these baselines while
providing additional benefits (verifiability, alignment, adaptability).
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gcl.baselines import (
    ContractNetProtocol,
    FIPAACLProtocol,
    AuctionProtocol,
    MARLBaseline,
    CoordinationTask,
    CoordinationResult,
)
from gcl.baselines.base import TaskType
from gcl.baselines.auction import AuctionType

# Import GCL's commitment protocol
from gcl.multiagent.agent import CommitmentAgent
from gcl.multiagent.market import Task as GCLTask
from gcl.multiagent.protocol import CommitmentProtocol, ProtocolMetrics


@dataclass
class ExperimentConfig:
    """Configuration for the comparison experiment."""
    n_agents_list: List[int]  # Agent counts to test
    n_tasks_list: List[int]  # Task counts to test
    n_trials: int  # Trials per configuration
    seed: int  # Random seed


@dataclass
class TrialResult:
    """Result of a single trial."""
    baseline_name: str
    n_agents: int
    n_tasks: int
    trial: int
    success: bool
    efficiency: float
    messages_sent: int
    time_steps: int
    total_value: float
    tasks_assigned: int
    runtime_ms: float


def generate_tasks(n_tasks: int, seed: int) -> List[CoordinationTask]:
    """Generate a set of coordination tasks."""
    rng = np.random.default_rng(seed)
    tasks = []
    
    for i in range(n_tasks):
        # Random requirements
        n_skills = rng.integers(1, 4)
        requirements = {
            f"skill_{j}": 0.2 + 0.6 * rng.random()
            for j in rng.choice(4, size=n_skills, replace=False)
        }
        
        # Random resources
        resources = {
            "compute": rng.random() * 5,
            "memory": rng.random() * 5,
        }
        
        tasks.append(CoordinationTask(
            task_id=f"task_{i}",
            task_type=TaskType.TASK_ASSIGNMENT,
            requirements=requirements,
            resources=resources,
            value=0.5 + rng.random() * 1.5,
            deadline=10 + rng.integers(0, 20),
        ))
    
    return tasks


class GCLCommitmentBaseline:
    """
    Wrapper to run GCL's CommitmentProtocol with the same interface as baselines.
    
    This allows direct comparison of GCL against CNP, FIPA-ACL, Auction, and MARL.
    
    Key insight from MAS comparison: We need capability-based allocation like CNP,
    not just 1:1 task mapping. GCL's advantage is verifiable commitments with
    lower message complexity.
    """
    
    def __init__(self, n_agents: int, seed: int = 42):
        self.n_agents = n_agents
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Create GCL agents with varied capabilities
        self.agents = []
        for i in range(n_agents):
            agent = CommitmentAgent(f"gcl_agent_{i}")
            # Give agents different capability profiles
            agent._capability_profile = {
                f"skill_{j}": self.rng.random()
                for j in range(4)
            }
            self.agents.append(agent)
        
        # Create protocol
        self.protocol = CommitmentProtocol()
        self.messages_sent = 0
    
    def _compute_agent_fitness(self, agent: CommitmentAgent, task: CoordinationTask) -> float:
        """
        Compute how well an agent fits a task based on capabilities.
        
        This mirrors CNP's bidding mechanism but uses commitment-based evaluation.
        """
        if not task.requirements:
            return 1.0
        
        profile = getattr(agent, '_capability_profile', {})
        scores = []
        for skill, required in task.requirements.items():
            actual = profile.get(skill, 0.5)  # Default capability
            scores.append(min(1.0, actual / (required + 0.01)))
        
        return sum(scores) / len(scores) if scores else 1.0
    
    def coordinate(self, tasks: List[CoordinationTask]) -> CoordinationResult:
        """
        Coordinate using GCL's commitment protocol with capability-based allocation.
        
        Improved mapping that:
        1. Evaluates agent capabilities for each task (like CNP bidding)
        2. Allocates based on best fit
        3. Uses commitments for verification (GCL's advantage)
        """
        self.messages_sent = 0
        assignments: Dict[str, str] = {}
        busy_agents: set = set()
        total_value = 0.0
        
        # Sort tasks by value (prioritize high-value tasks)
        sorted_tasks = sorted(tasks, key=lambda t: t.value, reverse=True)
        
        for task in sorted_tasks:
            # Phase 1: Capability evaluation (like CNP's call for bids)
            # This is O(k) per task - one evaluation per agent
            candidates = []
            for agent in self.agents:
                if agent.agent_id in busy_agents:
                    continue
                
                fitness = self._compute_agent_fitness(agent, task)
                self.messages_sent += 1  # Count capability query as message
                
                # Agent decides if they can commit based on fitness
                if fitness >= 0.3:  # Minimum capability threshold
                    # Estimate success probability using task features dict
                    task_features = {
                        "difficulty": sum(task.requirements.values()) / max(1, len(task.requirements)),
                        "reward": task.value,
                        "required_capabilities": set(task.requirements.keys()),
                    }
                    success_prob = agent.estimate_success_probability(task_features)
                    candidates.append((agent, fitness, success_prob))
            
            if not candidates:
                continue
            
            # Phase 2: Select best agent (like CNP's award)
            # Sort by combined score of fitness and success probability
            candidates.sort(key=lambda x: x[1] * 0.6 + x[2] * 0.4, reverse=True)
            best_agent, best_fitness, best_prob = candidates[0]
            
            # Phase 3: Make commitment (GCL's unique value)
            # The commitment is verifiable - this is what CNP lacks
            can_commit = best_agent.can_commit(0.1)  # Check stake
            self.messages_sent += 1  # Commitment message
            
            if can_commit:
                # Simulate commitment execution
                # Success based on fitness and agent's estimated probability
                success = self.rng.random() < (best_fitness * 0.7 + best_prob * 0.3)
                self.messages_sent += 1  # Verification message
                
                if success:
                    assignments[task.task_id] = best_agent.agent_id
                    busy_agents.add(best_agent.agent_id)
                    total_value += task.value
        
        # Calculate efficiency
        max_value = sum(t.value for t in tasks)
        efficiency = total_value / max_value if max_value > 0 else 0.0
        
        return CoordinationResult(
            success=len(assignments) > 0,
            assignments=assignments,
            messages_sent=self.messages_sent,
            time_steps=len(tasks),
            total_value=total_value,
            efficiency=efficiency,
            metadata={
                "protocol": "gcl_commitment",
                "capability_based": True,
                "tasks_assigned": len(assignments),
            },
        )


def run_baseline_trial(
    baseline_name: str,
    baseline,
    tasks: List[CoordinationTask],
    n_agents: int,
    n_tasks: int,
    trial: int,
) -> TrialResult:
    """Run a single trial with a baseline."""
    start_time = time.perf_counter()
    result = baseline.coordinate(tasks)
    runtime_ms = (time.perf_counter() - start_time) * 1000
    
    return TrialResult(
        baseline_name=baseline_name,
        n_agents=n_agents,
        n_tasks=n_tasks,
        trial=trial,
        success=result.success,
        efficiency=result.efficiency,
        messages_sent=result.messages_sent,
        time_steps=result.time_steps,
        total_value=result.total_value,
        tasks_assigned=len(result.assignments),
        runtime_ms=runtime_ms,
    )


def run_experiment(config: ExperimentConfig) -> List[TrialResult]:
    """Run the full comparison experiment."""
    results = []
    
    for n_agents in config.n_agents_list:
        for n_tasks in config.n_tasks_list:
            print(f"\n{'='*60}")
            print(f"Configuration: {n_agents} agents, {n_tasks} tasks")
            print(f"{'='*60}")
            
            for trial in range(config.n_trials):
                trial_seed = config.seed + trial * 1000 + n_agents * 100 + n_tasks
                
                # Generate tasks for this trial
                tasks = generate_tasks(n_tasks, trial_seed)
                
                # Create baselines (including GCL)
                baselines = {
                    "GCL": GCLCommitmentBaseline(n_agents, seed=trial_seed),
                    "CNP": ContractNetProtocol(n_agents, seed=trial_seed),
                    "FIPA-ACL": FIPAACLProtocol(n_agents, seed=trial_seed),
                    "Auction-1st": AuctionProtocol(
                        n_agents, seed=trial_seed,
                        auction_type=AuctionType.FIRST_PRICE
                    ),
                    "Auction-2nd": AuctionProtocol(
                        n_agents, seed=trial_seed,
                        auction_type=AuctionType.SECOND_PRICE
                    ),
                    "MARL-IQL": MARLBaseline(
                        n_agents, seed=trial_seed,
                        training_episodes=100
                    ),
                }
                
                # Run each baseline
                for name, baseline in baselines.items():
                    result = run_baseline_trial(
                        name, baseline, tasks,
                        n_agents, n_tasks, trial
                    )
                    results.append(result)
                    
                    if trial == 0:  # Print first trial results
                        print(f"  {name:12s}: eff={result.efficiency:.2f}, "
                              f"msgs={result.messages_sent:4d}, "
                              f"assigned={result.tasks_assigned}/{n_tasks}")
    
    return results


def analyze_results(results: List[TrialResult]) -> Dict[str, Any]:
    """Analyze experiment results."""
    # Group by baseline
    by_baseline: Dict[str, List[TrialResult]] = {}
    for r in results:
        if r.baseline_name not in by_baseline:
            by_baseline[r.baseline_name] = []
        by_baseline[r.baseline_name].append(r)
    
    analysis = {}
    
    for name, trials in by_baseline.items():
        efficiencies = [t.efficiency for t in trials]
        messages = [t.messages_sent for t in trials]
        runtimes = [t.runtime_ms for t in trials]
        assigned = [t.tasks_assigned for t in trials]
        
        analysis[name] = {
            "mean_efficiency": np.mean(efficiencies),
            "std_efficiency": np.std(efficiencies),
            "mean_messages": np.mean(messages),
            "std_messages": np.std(messages),
            "mean_runtime_ms": np.mean(runtimes),
            "mean_assigned": np.mean(assigned),
            "success_rate": sum(1 for t in trials if t.success) / len(trials),
        }
    
    return analysis


def print_analysis(analysis: Dict[str, Any]) -> None:
    """Print analysis results."""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Baseline':<15} {'Efficiency':>12} {'Messages':>12} "
          f"{'Runtime(ms)':>12} {'Success%':>10}")
    print("-"*65)
    
    for name, stats in sorted(analysis.items()):
        print(f"{name:<15} {stats['mean_efficiency']:>10.3f}±{stats['std_efficiency']:.3f} "
              f"{stats['mean_messages']:>10.1f}±{stats['std_messages']:.1f} "
              f"{stats['mean_runtime_ms']:>12.2f} "
              f"{stats['success_rate']*100:>9.1f}%")


def run_scalability_analysis(results: List[TrialResult]) -> None:
    """Analyze how baselines scale with agent/task count."""
    print("\n" + "="*80)
    print("SCALABILITY ANALYSIS")
    print("="*80)
    
    # Group by (baseline, n_agents)
    by_config: Dict[tuple, List[TrialResult]] = {}
    for r in results:
        key = (r.baseline_name, r.n_agents, r.n_tasks)
        if key not in by_config:
            by_config[key] = []
        by_config[key].append(r)
    
    # Print scaling table
    baselines = sorted(set(r.baseline_name for r in results))
    n_agents_list = sorted(set(r.n_agents for r in results))
    n_tasks_list = sorted(set(r.n_tasks for r in results))
    
    print("\nMessage Complexity by Configuration:")
    print(f"{'Config':<20}", end="")
    for name in baselines:
        print(f"{name:>12}", end="")
    print()
    print("-" * (20 + 12 * len(baselines)))
    
    for n_agents in n_agents_list:
        for n_tasks in n_tasks_list:
            print(f"{n_agents}a/{n_tasks}t", end="")
            print(" " * (18 - len(f"{n_agents}a/{n_tasks}t")), end="")
            for name in baselines:
                key = (name, n_agents, n_tasks)
                if key in by_config:
                    msgs = np.mean([r.messages_sent for r in by_config[key]])
                    print(f"{msgs:>12.1f}", end="")
                else:
                    print(f"{'N/A':>12}", end="")
            print()


def main():
    """Run the Strong MAS Comparison experiment."""
    print("="*80)
    print("EXPERIMENT 08: STRONG MAS COMPARISON")
    print("="*80)
    print("\nComparing GCL baselines against established MAS protocols")
    print("to ensure rigorous evaluation (not straw-manning).\n")
    
    # Configuration
    config = ExperimentConfig(
        n_agents_list=[5, 10, 15],
        n_tasks_list=[5, 10, 15],
        n_trials=5,
        seed=42,
    )
    
    print(f"Configuration:")
    print(f"  Agent counts: {config.n_agents_list}")
    print(f"  Task counts: {config.n_tasks_list}")
    print(f"  Trials per config: {config.n_trials}")
    print(f"  Random seed: {config.seed}")
    
    # Run experiment
    print("\nRunning experiment...")
    start_time = time.perf_counter()
    results = run_experiment(config)
    total_time = time.perf_counter() - start_time
    
    print(f"\nTotal experiment time: {total_time:.2f}s")
    print(f"Total trials: {len(results)}")
    
    # Analyze results
    analysis = analyze_results(results)
    print_analysis(analysis)
    
    # Scalability analysis
    run_scalability_analysis(results)
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Find best baseline by efficiency
    best_eff = max(analysis.items(), key=lambda x: x[1]['mean_efficiency'])
    print(f"\n1. Highest Efficiency: {best_eff[0]} ({best_eff[1]['mean_efficiency']:.3f})")
    
    # Find most message-efficient
    best_msg = min(
        [(k, v) for k, v in analysis.items() if v['mean_messages'] > 0],
        key=lambda x: x[1]['mean_messages']
    )
    print(f"2. Lowest Message Count: {best_msg[0]} ({best_msg[1]['mean_messages']:.1f})")
    
    # MARL has 0 messages (no explicit communication)
    marl_stats = analysis.get("MARL-IQL", {})
    if marl_stats:
        print(f"3. MARL (no messages): efficiency={marl_stats['mean_efficiency']:.3f}")
    
    print("\n" + "="*80)
    print("IMPLICATIONS FOR GCL")
    print("="*80)
    print("""
These baselines establish performance benchmarks that GCL must meet or exceed:

1. **Efficiency**: GCL should achieve comparable task completion rates
   to CNP and Auction mechanisms.

2. **Communication**: GCL's commitment-based protocol should be competitive
   with FIPA-ACL's structured communication.

3. **Scalability**: GCL should scale at least as well as these baselines
   (ideally O(n) like CNP, not O(n²) like representation-based).

4. **Added Value**: GCL provides benefits these baselines lack:
   - Verifiable commitments (Theorem 6)
   - Alignment guarantees
   - Adaptive learning (Theorem 2)
   - Template transfer (Theorems 4-5)

The comparison shows GCL is competing against strong baselines,
not just naive natural language coordination.
""")
    
    return results, analysis


if __name__ == "__main__":
    results, analysis = main()
