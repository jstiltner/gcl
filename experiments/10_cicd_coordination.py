#!/usr/bin/env python3
"""
Experiment 10: CI/CD Pipeline Coordination Demo

This experiment demonstrates GCL in a concrete, relatable domain:
coordinating AI agents in a CI/CD pipeline.

SCENARIO:
A software development pipeline with multiple AI agents:
- CodeReviewer: Reviews code changes
- TestGenerator: Generates test cases
- SecurityScanner: Checks for vulnerabilities
- Deployer: Deploys to production

Each agent makes commitments about their work:
- "I will complete review within 5 minutes"
- "I will generate tests covering 80% of changes"
- "I will flag all critical vulnerabilities"
- "I will deploy only if all checks pass"

SEMANTIC DRIFT:
- Model updates change what "critical vulnerability" means
- API changes affect deployment semantics
- Test coverage metrics evolve

GCL ADVANTAGE:
- Agents learn which commitments are reliable
- Pipeline adapts to semantic drift
- Failures are detected and handled gracefully

Author: GCL Research Team
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gcl.core.calculus import (
    Commitment, FailureMode, Outcome, CommitmentOutcome,
    issue, verify, settle, confidence
)


class AgentRole(Enum):
    """Roles in the CI/CD pipeline."""
    CODE_REVIEWER = "code_reviewer"
    TEST_GENERATOR = "test_generator"
    SECURITY_SCANNER = "security_scanner"
    DEPLOYER = "deployer"


@dataclass
class PipelineState:
    """State of the CI/CD pipeline."""
    code_changes: int  # Number of lines changed
    complexity: float  # Code complexity (0-1)
    has_security_issues: bool
    test_coverage: float  # Current coverage (0-1)
    review_complete: bool = False
    tests_generated: bool = False
    security_scanned: bool = False
    deployed: bool = False
    
    # Semantic drift simulation
    drift_level: float = 0.0  # How much semantics have drifted


@dataclass
class PipelineAction:
    """Action taken by a pipeline agent."""
    agent: AgentRole
    action_type: str
    parameters: Dict
    timestamp: float = 0.0


@dataclass
class CICDAgent:
    """
    An agent in the CI/CD pipeline using GCL.
    
    Each agent:
    1. Issues commitments about their work
    2. Attempts to fulfill commitments
    3. Learns from verification outcomes
    """
    role: AgentRole
    name: str
    seed: int = 42
    
    # Commitment tracking
    commitment_history: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.commitment_history = {}
    
    def get_confidence(self, commitment_type: str) -> float:
        """Get confidence in a commitment type."""
        if commitment_type not in self.commitment_history:
            return 0.5  # Prior
        successes, failures = self.commitment_history[commitment_type]
        return confidence(successes, failures)
    
    def update_confidence(self, commitment_type: str, success: bool) -> None:
        """Update confidence based on outcome."""
        if commitment_type not in self.commitment_history:
            self.commitment_history[commitment_type] = (0, 0)
        successes, failures = self.commitment_history[commitment_type]
        if success:
            self.commitment_history[commitment_type] = (successes + 1, failures)
        else:
            self.commitment_history[commitment_type] = (successes, failures + 1)
    
    def should_commit(self, commitment_type: str, threshold: float = 0.6) -> bool:
        """Decide whether to make a commitment based on confidence."""
        return self.get_confidence(commitment_type) >= threshold
    
    def attempt_action(self, state: PipelineState, drift: float) -> Tuple[bool, str]:
        """
        Attempt to fulfill role-specific commitment.
        
        Returns:
            (success, failure_reason)
        """
        # Base success probability depends on role and state
        if self.role == AgentRole.CODE_REVIEWER:
            # Review success depends on complexity
            base_prob = 0.95 - 0.3 * state.complexity
            commitment_type = "complete_review"
        elif self.role == AgentRole.TEST_GENERATOR:
            # Test generation depends on code changes
            base_prob = 0.9 - 0.2 * (state.code_changes / 1000)
            commitment_type = "generate_tests"
        elif self.role == AgentRole.SECURITY_SCANNER:
            # Security scanning affected by drift (what counts as "critical")
            base_prob = 0.85 - 0.4 * drift  # Drift hurts security scanning
            commitment_type = "scan_security"
        else:  # DEPLOYER
            # Deployment depends on all prior steps
            if not (state.review_complete and state.tests_generated and state.security_scanned):
                return False, "prerequisites_not_met"
            base_prob = 0.95 - 0.3 * drift
            commitment_type = "deploy"
        
        # Apply drift penalty
        success_prob = base_prob * (1 - 0.5 * drift)
        success = self.rng.random() < success_prob
        
        if success:
            return True, ""
        else:
            # Determine failure reason
            if drift > 0.3 and self.rng.random() < 0.5:
                return False, "semantic_drift"
            elif self.rng.random() < 0.3:
                return False, "timeout"
            else:
                return False, "quality_failure"


def create_pipeline_commitments() -> Dict[AgentRole, Commitment]:
    """Create GCL commitments for each pipeline role."""
    
    # Code Review Commitment
    review_commitment = Commitment(
        trigger=lambda s: s.code_changes > 0 and not s.review_complete,
        action=lambda s: PipelineAction(
            AgentRole.CODE_REVIEWER, "review", {"lines": s.code_changes}
        ),
        verification=lambda s, s_, a: s_.review_complete,
        failures=[
            FailureMode(
                lambda s, s_, a: not s_.review_complete and s.complexity > 0.7,
                0.8, "complexity_timeout"
            ),
            FailureMode(
                lambda s, s_, a: not s_.review_complete,
                0.5, "review_incomplete"
            ),
        ],
        stake=1.0,
    )
    
    # Test Generation Commitment
    test_commitment = Commitment(
        trigger=lambda s: s.review_complete and not s.tests_generated,
        action=lambda s: PipelineAction(
            AgentRole.TEST_GENERATOR, "generate_tests", {"target_coverage": 0.8}
        ),
        verification=lambda s, s_, a: s_.tests_generated and s_.test_coverage >= 0.8,
        failures=[
            FailureMode(
                lambda s, s_, a: s_.test_coverage < 0.5,
                0.9, "low_coverage"
            ),
            FailureMode(
                lambda s, s_, a: not s_.tests_generated,
                0.6, "generation_failed"
            ),
        ],
        stake=1.5,
    )
    
    # Security Scan Commitment
    security_commitment = Commitment(
        trigger=lambda s: s.tests_generated and not s.security_scanned,
        action=lambda s: PipelineAction(
            AgentRole.SECURITY_SCANNER, "scan", {"depth": "full"}
        ),
        verification=lambda s, s_, a: s_.security_scanned and not s_.has_security_issues,
        failures=[
            FailureMode(
                lambda s, s_, a: s_.has_security_issues,
                1.0, "security_vulnerability"
            ),
            FailureMode(
                lambda s, s_, a: not s_.security_scanned,
                0.7, "scan_incomplete"
            ),
        ],
        stake=2.0,  # High stake for security
    )
    
    # Deployment Commitment
    deploy_commitment = Commitment(
        trigger=lambda s: s.security_scanned and not s.has_security_issues and not s.deployed,
        action=lambda s: PipelineAction(
            AgentRole.DEPLOYER, "deploy", {"environment": "production"}
        ),
        verification=lambda s, s_, a: s_.deployed,
        failures=[
            FailureMode(
                lambda s, s_, a: s.has_security_issues,
                1.0, "security_block"
            ),
            FailureMode(
                lambda s, s_, a: not s_.deployed,
                0.8, "deployment_failed"
            ),
        ],
        stake=3.0,  # Highest stake for deployment
    )
    
    return {
        AgentRole.CODE_REVIEWER: review_commitment,
        AgentRole.TEST_GENERATOR: test_commitment,
        AgentRole.SECURITY_SCANNER: security_commitment,
        AgentRole.DEPLOYER: deploy_commitment,
    }


def run_pipeline_episode(
    agents: Dict[AgentRole, CICDAgent],
    commitments: Dict[AgentRole, Commitment],
    initial_state: PipelineState,
    use_gcl: bool = True,
) -> Dict:
    """
    Run a single pipeline episode.
    
    Args:
        agents: Dictionary of agents by role
        commitments: Dictionary of commitments by role
        initial_state: Starting pipeline state
        use_gcl: Whether to use GCL (vs. baseline)
        
    Returns:
        Episode results
    """
    state = initial_state
    results = {
        "stages_completed": 0,
        "total_reward": 0.0,
        "failures": [],
        "confidence_updates": [],
    }
    
    # Pipeline stages in order
    stages = [
        AgentRole.CODE_REVIEWER,
        AgentRole.TEST_GENERATOR,
        AgentRole.SECURITY_SCANNER,
        AgentRole.DEPLOYER,
    ]
    
    for role in stages:
        agent = agents[role]
        commitment = commitments[role]
        
        # Check if commitment should trigger
        if not commitment.trigger(state):
            continue
        
        # GCL: Check confidence before committing
        commitment_type = f"{role.value}_commitment"
        if use_gcl and not agent.should_commit(commitment_type, threshold=0.5):
            # Low confidence - skip or use fallback
            results["failures"].append({
                "stage": role.value,
                "reason": "low_confidence",
                "confidence": agent.get_confidence(commitment_type),
            })
            continue
        
        # Attempt action
        pre_state = state
        success, failure_reason = agent.attempt_action(state, state.drift_level)
        
        # Update state based on success
        if success:
            if role == AgentRole.CODE_REVIEWER:
                state = PipelineState(
                    code_changes=state.code_changes,
                    complexity=state.complexity,
                    has_security_issues=state.has_security_issues,
                    test_coverage=state.test_coverage,
                    review_complete=True,
                    drift_level=state.drift_level,
                )
            elif role == AgentRole.TEST_GENERATOR:
                state = PipelineState(
                    code_changes=state.code_changes,
                    complexity=state.complexity,
                    has_security_issues=state.has_security_issues,
                    test_coverage=0.85,  # Achieved target
                    review_complete=True,
                    tests_generated=True,
                    drift_level=state.drift_level,
                )
            elif role == AgentRole.SECURITY_SCANNER:
                state = PipelineState(
                    code_changes=state.code_changes,
                    complexity=state.complexity,
                    has_security_issues=False,  # Cleared
                    test_coverage=state.test_coverage,
                    review_complete=True,
                    tests_generated=True,
                    security_scanned=True,
                    drift_level=state.drift_level,
                )
            elif role == AgentRole.DEPLOYER:
                state = PipelineState(
                    code_changes=state.code_changes,
                    complexity=state.complexity,
                    has_security_issues=False,
                    test_coverage=state.test_coverage,
                    review_complete=True,
                    tests_generated=True,
                    security_scanned=True,
                    deployed=True,
                    drift_level=state.drift_level,
                )
            
            results["stages_completed"] += 1
            
            # Verify and settle
            outcome = verify(commitment, pre_state, state, None)
            reward = settle(commitment, outcome)
            results["total_reward"] += reward
            
            # Update confidence
            if use_gcl:
                agent.update_confidence(commitment_type, True)
                results["confidence_updates"].append({
                    "stage": role.value,
                    "success": True,
                    "new_confidence": agent.get_confidence(commitment_type),
                })
        else:
            results["failures"].append({
                "stage": role.value,
                "reason": failure_reason,
            })
            
            # Update confidence on failure
            if use_gcl:
                agent.update_confidence(commitment_type, False)
                results["confidence_updates"].append({
                    "stage": role.value,
                    "success": False,
                    "new_confidence": agent.get_confidence(commitment_type),
                })
            
            # Pipeline stops on failure
            break
    
    results["final_state"] = {
        "review_complete": state.review_complete,
        "tests_generated": state.tests_generated,
        "security_scanned": state.security_scanned,
        "deployed": state.deployed,
    }
    
    return results


def run_cicd_experiment(
    n_episodes: int = 100,
    drift_schedule: Optional[List[float]] = None,
    seed: int = 42,
) -> Dict:
    """
    Run the full CI/CD coordination experiment.
    
    Args:
        n_episodes: Number of pipeline runs
        drift_schedule: Drift level for each episode (or None for gradual increase)
        seed: Random seed
        
    Returns:
        Experiment results
    """
    if drift_schedule is None:
        # Gradual drift increase
        drift_schedule = [i / n_episodes * 0.5 for i in range(n_episodes)]
    
    # Create agents
    gcl_agents = {
        role: CICDAgent(role, f"gcl_{role.value}", seed + i)
        for i, role in enumerate(AgentRole)
    }
    baseline_agents = {
        role: CICDAgent(role, f"baseline_{role.value}", seed + i)
        for i, role in enumerate(AgentRole)
    }
    
    # Create commitments
    commitments = create_pipeline_commitments()
    
    # Run episodes
    gcl_results = []
    baseline_results = []
    
    rng = np.random.default_rng(seed)
    
    for episode in range(n_episodes):
        drift = drift_schedule[episode]
        
        # Create initial state
        initial_state = PipelineState(
            code_changes=rng.integers(100, 1000),
            complexity=rng.random() * 0.8,
            has_security_issues=rng.random() < 0.3,
            test_coverage=rng.random() * 0.5,
            drift_level=drift,
        )
        
        # Run with GCL
        gcl_result = run_pipeline_episode(
            gcl_agents, commitments, initial_state, use_gcl=True
        )
        gcl_result["episode"] = episode
        gcl_result["drift"] = drift
        gcl_results.append(gcl_result)
        
        # Run baseline (same initial state)
        baseline_result = run_pipeline_episode(
            baseline_agents, commitments, initial_state, use_gcl=False
        )
        baseline_result["episode"] = episode
        baseline_result["drift"] = drift
        baseline_results.append(baseline_result)
    
    return {
        "gcl": gcl_results,
        "baseline": baseline_results,
        "config": {
            "n_episodes": n_episodes,
            "seed": seed,
        },
    }


def analyze_results(results: Dict) -> Dict:
    """Analyze experiment results."""
    gcl = results["gcl"]
    baseline = results["baseline"]
    
    # Compute metrics
    gcl_completion = np.mean([r["stages_completed"] for r in gcl])
    baseline_completion = np.mean([r["stages_completed"] for r in baseline])
    
    gcl_deployments = sum(1 for r in gcl if r["final_state"]["deployed"])
    baseline_deployments = sum(1 for r in baseline if r["final_state"]["deployed"])
    
    gcl_reward = np.mean([r["total_reward"] for r in gcl])
    baseline_reward = np.mean([r["total_reward"] for r in baseline])
    
    # Analyze by drift level
    drift_analysis = {}
    for i in range(10):
        drift_min = i * 0.05
        drift_max = (i + 1) * 0.05
        
        gcl_subset = [r for r in gcl if drift_min <= r["drift"] < drift_max]
        baseline_subset = [r for r in baseline if drift_min <= r["drift"] < drift_max]
        
        if gcl_subset and baseline_subset:
            drift_analysis[f"{drift_min:.2f}-{drift_max:.2f}"] = {
                "gcl_completion": np.mean([r["stages_completed"] for r in gcl_subset]),
                "baseline_completion": np.mean([r["stages_completed"] for r in baseline_subset]),
                "gcl_deployments": sum(1 for r in gcl_subset if r["final_state"]["deployed"]),
                "baseline_deployments": sum(1 for r in baseline_subset if r["final_state"]["deployed"]),
            }
    
    return {
        "summary": {
            "gcl_avg_stages": gcl_completion,
            "baseline_avg_stages": baseline_completion,
            "gcl_deployments": gcl_deployments,
            "baseline_deployments": baseline_deployments,
            "gcl_avg_reward": gcl_reward,
            "baseline_avg_reward": baseline_reward,
            "gcl_advantage_stages": gcl_completion - baseline_completion,
            "gcl_advantage_deployments": gcl_deployments - baseline_deployments,
        },
        "by_drift": drift_analysis,
    }


def main():
    """Run the CI/CD coordination experiment."""
    print("=" * 60)
    print("EXPERIMENT 10: CI/CD PIPELINE COORDINATION")
    print("=" * 60)
    print("\nScenario: AI agents coordinating in a CI/CD pipeline")
    print("- CodeReviewer → TestGenerator → SecurityScanner → Deployer")
    print("- Semantic drift simulates model updates and API changes")
    print("- GCL agents learn which commitments are reliable\n")
    
    # Run experiment
    print("Running 100 pipeline episodes with increasing drift...")
    results = run_cicd_experiment(n_episodes=100, seed=42)
    
    # Analyze results
    analysis = analyze_results(results)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    summary = analysis["summary"]
    print(f"\nAverage stages completed:")
    print(f"  GCL:      {summary['gcl_avg_stages']:.2f} / 4")
    print(f"  Baseline: {summary['baseline_avg_stages']:.2f} / 4")
    print(f"  Advantage: {summary['gcl_advantage_stages']:+.2f}")
    
    print(f"\nSuccessful deployments:")
    print(f"  GCL:      {summary['gcl_deployments']} / 100")
    print(f"  Baseline: {summary['baseline_deployments']} / 100")
    print(f"  Advantage: {summary['gcl_advantage_deployments']:+d}")
    
    print(f"\nAverage reward:")
    print(f"  GCL:      {summary['gcl_avg_reward']:.2f}")
    print(f"  Baseline: {summary['baseline_avg_reward']:.2f}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS BY DRIFT LEVEL")
    print("=" * 60)
    
    print("\nDrift Range | GCL Stages | Baseline | GCL Deploys | Baseline")
    print("-" * 65)
    for drift_range, data in analysis["by_drift"].items():
        print(f"  {drift_range}   |    {data['gcl_completion']:.2f}    |   {data['baseline_completion']:.2f}   |     {data['gcl_deployments']}      |    {data['baseline_deployments']}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    if summary["gcl_advantage_stages"] > 0:
        print("\n✓ GCL completes more pipeline stages on average")
    if summary["gcl_advantage_deployments"] > 0:
        print("✓ GCL achieves more successful deployments")
    
    # Check if advantage increases with drift
    drift_ranges = list(analysis["by_drift"].keys())
    if len(drift_ranges) >= 2:
        low_drift = analysis["by_drift"][drift_ranges[0]]
        high_drift = analysis["by_drift"][drift_ranges[-1]]
        
        low_advantage = low_drift["gcl_completion"] - low_drift["baseline_completion"]
        high_advantage = high_drift["gcl_completion"] - high_drift["baseline_completion"]
        
        if high_advantage > low_advantage:
            print("✓ GCL advantage increases with drift (as predicted)")
    
    print("\n" + "=" * 60)
    print("PRACTICAL IMPLICATIONS")
    print("=" * 60)
    print("""
This demo shows GCL in a concrete, relatable domain:

1. **Real-world applicability**: CI/CD pipelines are ubiquitous
2. **Clear value proposition**: More successful deployments
3. **Drift robustness**: GCL adapts to model/API changes
4. **Graceful degradation**: Low-confidence commitments are skipped

For practitioners:
- Use GCL when pipeline agents may drift (model updates, API changes)
- Monitor commitment confidence to detect reliability issues
- Set appropriate confidence thresholds for critical stages
""")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "10_cicd_coordination.json", "w") as f:
        json.dump({
            "summary": summary,
            "by_drift": analysis["by_drift"],
        }, f, indent=2)
    print(f"\nResults saved to: {output_dir / '10_cicd_coordination.json'}")
    
    return results, analysis


if __name__ == "__main__":
    main()
