"""
Capitalist Arena - the competitive environment where firms battle.

Firms compete on tasks, winners gain resources, losers lose resources.
Bankruptcy eliminates firms from competition.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from collections import defaultdict

from .firm import Firm, FirmTaskOutcome
from ..agents.agent import Task
from ..config.capitalism import CapitalistMetaNetwork, MarketConditions


@dataclass
class TaskCompetitionResult:
    """Result of firms competing on a single task."""
    task: Task
    winner: Optional[Firm]
    attempts: List[FirmTaskOutcome]
    reward_distributed: float = 0.0


@dataclass
class RoundResult:
    """Result of one round of competition."""
    round_num: int
    task_results: List[TaskCompetitionResult]
    deaths: List[Firm]
    survivors: List[Firm]
    
    @property
    def n_tasks(self) -> int:
        return len(self.task_results)
    
    @property
    def n_deaths(self) -> int:
        return len(self.deaths)


@dataclass
class SimulationResult:
    """Result of full simulation."""
    rounds: int
    history: List[RoundResult]
    survivors: List[Firm]
    winner: Optional[Firm]
    
    def get_survivor_counts_by_type(self) -> Dict[str, int]:
        """Count survivors by structure type."""
        counts = defaultdict(int)
        for firm in self.survivors:
            counts[firm.structure.name] += 1
        return dict(counts)
    
    def get_survivor_resources_by_type(self) -> Dict[str, float]:
        """Sum resources by structure type."""
        resources = defaultdict(float)
        for firm in self.survivors:
            resources[firm.structure.name] += firm.resources
        return dict(resources)


class CapitalistArena:
    """
    The competitive environment where firms battle.
    
    Each round:
    1. Generate tasks
    2. Firms compete on each task
    3. Winners gain resources, losers lose resources
    4. Bankrupt firms are eliminated
    """
    
    def __init__(self, 
                 config: CapitalistMetaNetwork = None, 
                 market: MarketConditions = None):
        self.config = config or CapitalistMetaNetwork()
        self.market = market or MarketConditions()
        self.firms: List[Firm] = []
        self.round: int = 0
        self.history: List[RoundResult] = []
        self.dead_firms: List[Firm] = []
    
    def add_firm(self, firm: Firm):
        """Add a firm to the competition."""
        self.firms.append(firm)
    
    def generate_tasks(self) -> List[Task]:
        """Generate tasks for this round."""
        tasks = []
        for i in range(self.market.tasks_per_round):
            difficulty = random.gauss(
                self.market.task_difficulty_mean, 
                self.market.task_difficulty_variance
            )
            difficulty = max(0.1, min(0.9, difficulty))
            
            # Reward scales with difficulty
            reward = 1.0 + difficulty * 2.0
            
            tasks.append(Task(
                id=f"task_{self.round}_{i}",
                difficulty=difficulty,
                reward=reward
            ))
        return tasks
    
    def run_competition(self, task: Task) -> TaskCompetitionResult:
        """Firms compete on a single task."""
        
        # Each firm attempts
        attempts = []
        for firm in self.firms:
            if firm.resources > 0:  # Can afford to compete
                firm.run_internal_round()
                outcome = firm.attempt_task(task)
                attempts.append(outcome)
        
        if not attempts:
            return TaskCompetitionResult(task=task, winner=None, attempts=[])
        
        # Determine winner (highest output)
        winner_outcome = max(attempts, key=lambda a: a.output)
        winner = winner_outcome.firm if winner_outcome.output > 0 else None
        
        # Distribute rewards
        reward_distributed = 0.0
        if winner:
            if self.market.winner_take_all > 0.5:
                # Winner takes most
                winner_reward = task.reward * self.market.winner_take_all
                winner.add_resources(winner_reward)
                reward_distributed = winner_reward
                
                for attempt in attempts:
                    if attempt.firm != winner:
                        attempt.firm.remove_resources(self.config.loser_cost)
            else:
                # Proportional distribution
                total_output = sum(a.output for a in attempts)
                if total_output > 0:
                    for attempt in attempts:
                        share = attempt.output / total_output
                        reward = task.reward * share
                        attempt.firm.add_resources(reward)
                        reward_distributed += reward
        
        # End internal round for all firms
        for firm in self.firms:
            firm.end_internal_round()
        
        return TaskCompetitionResult(
            task=task,
            winner=winner,
            attempts=attempts,
            reward_distributed=reward_distributed
        )
    
    def run_round(self) -> RoundResult:
        """One round of competition."""
        self.round += 1
        
        # Generate tasks for this round
        tasks = self.generate_tasks()
        
        # Firms compete on each task
        task_results = []
        for task in tasks:
            result = self.run_competition(task)
            task_results.append(result)
        
        # Check for bankruptcies
        survivors = []
        deaths = []
        for firm in self.firms:
            if firm.is_bankrupt():
                deaths.append(firm)
                self.dead_firms.append(firm)
            else:
                survivors.append(firm)
        
        self.firms = survivors
        
        result = RoundResult(
            round_num=self.round,
            task_results=task_results,
            deaths=deaths,
            survivors=survivors
        )
        self.history.append(result)
        return result
    
    def run_simulation(self, n_rounds: int) -> SimulationResult:
        """Run full simulation."""
        for _ in range(n_rounds):
            if len(self.firms) <= 1:
                break  # Competition over
            self.run_round()
        
        winner = None
        if self.firms:
            winner = max(self.firms, key=lambda f: f.resources)
        
        return SimulationResult(
            rounds=self.round,
            history=self.history,
            survivors=self.firms,
            winner=winner
        )
    
    def get_firm_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get current metrics for all firms."""
        metrics = {}
        for firm in self.firms:
            metrics[firm.id] = {
                "structure": firm.structure.name,
                "resources": firm.resources,
                "wins": firm.wins,
                "losses": firm.losses,
                "avg_reputation": firm.get_average_reputation(),
                "avg_capability": firm.get_average_capability(),
                "gini": firm.get_gini_coefficient(),
                "total_templates": firm.get_total_templates(),
                "unique_templates": firm.get_unique_templates(),
                "status_distribution": firm.get_status_distribution(),
            }
        return metrics
    
    def get_structure_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary metrics by structure type."""
        by_structure = defaultdict(list)
        for firm in self.firms:
            by_structure[firm.structure.name].append(firm)
        
        summary = {}
        for structure_name, firms in by_structure.items():
            summary[structure_name] = {
                "n_firms": len(firms),
                "total_resources": sum(f.resources for f in firms),
                "avg_resources": sum(f.resources for f in firms) / len(firms) if firms else 0,
                "total_wins": sum(f.wins for f in firms),
                "total_losses": sum(f.losses for f in firms),
                "avg_gini": sum(f.get_gini_coefficient() for f in firms) / len(firms) if firms else 0,
                "avg_capability": sum(f.get_average_capability() for f in firms) / len(firms) if firms else 0,
            }
        return summary
