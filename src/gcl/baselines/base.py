"""
Base classes for MAS baseline implementations.

Provides common interfaces and data structures for comparing
different coordination mechanisms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class TaskType(str, Enum):
    """Types of coordination tasks."""
    RESOURCE_ALLOCATION = "resource_allocation"
    TASK_ASSIGNMENT = "task_assignment"
    COALITION_FORMATION = "coalition_formation"
    NEGOTIATION = "negotiation"


@dataclass
class CoordinationTask:
    """
    A coordination task for MAS comparison.
    
    Attributes:
        task_id: Unique identifier.
        task_type: Type of coordination required.
        resources: Resources to allocate or compete for.
        requirements: Task requirements (capabilities needed).
        deadline: Optional deadline in timesteps.
        value: Value/reward for completing the task.
    """
    task_id: str = field(default_factory=lambda: str(uuid4()))
    task_type: TaskType = TaskType.TASK_ASSIGNMENT
    resources: Dict[str, float] = field(default_factory=dict)
    requirements: Dict[str, float] = field(default_factory=dict)
    deadline: Optional[int] = None
    value: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapability:
    """
    An agent's capability profile.
    
    Attributes:
        agent_id: Unique identifier.
        capabilities: Dict of capability name to level (0-1).
        resources: Available resources.
        cost: Cost per unit of work.
    """
    agent_id: str
    capabilities: Dict[str, float] = field(default_factory=dict)
    resources: Dict[str, float] = field(default_factory=dict)
    cost: float = 1.0


@dataclass
class CoordinationResult:
    """
    Result of a coordination attempt.
    
    Attributes:
        success: Whether coordination succeeded.
        assignments: Dict of task_id -> agent_id assignments.
        messages_sent: Number of messages exchanged.
        time_steps: Time steps to reach coordination.
        total_value: Total value achieved.
        efficiency: Value achieved / optimal value.
    """
    success: bool
    assignments: Dict[str, str] = field(default_factory=dict)
    messages_sent: int = 0
    time_steps: int = 0
    total_value: float = 0.0
    efficiency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MASBaseline(ABC):
    """
    Abstract base class for MAS coordination baselines.
    
    All baselines must implement the coordinate() method which
    takes a task and agent capabilities and returns a coordination result.
    """
    
    def __init__(self, n_agents: int, seed: int = 42):
        """
        Initialize the baseline.
        
        Args:
            n_agents: Number of agents.
            seed: Random seed for reproducibility.
        """
        self.n_agents = n_agents
        self.seed = seed
        self.agents: List[AgentCapability] = []
        self._initialize_agents()
    
    def _initialize_agents(self) -> None:
        """Initialize agent capabilities."""
        import numpy as np
        np.random.seed(self.seed)
        
        self.agents = []
        for i in range(self.n_agents):
            # Random capabilities
            capabilities = {
                f"skill_{j}": np.random.random()
                for j in range(4)
            }
            resources = {
                "compute": np.random.random() * 10,
                "memory": np.random.random() * 10,
            }
            self.agents.append(AgentCapability(
                agent_id=f"agent_{i}",
                capabilities=capabilities,
                resources=resources,
                cost=0.5 + np.random.random(),
            ))
    
    @abstractmethod
    def coordinate(
        self,
        tasks: List[CoordinationTask],
    ) -> CoordinationResult:
        """
        Coordinate agents to complete tasks.
        
        Args:
            tasks: List of tasks to coordinate.
            
        Returns:
            CoordinationResult with assignments and metrics.
        """
        pass
    
    def get_agent_capability(self, agent_id: str, skill: str) -> float:
        """Get an agent's capability level for a skill."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent.capabilities.get(skill, 0.0)
        return 0.0
    
    def compute_task_fitness(
        self,
        agent: AgentCapability,
        task: CoordinationTask,
    ) -> float:
        """
        Compute how well an agent fits a task.
        
        Args:
            agent: Agent capability profile.
            task: Task to evaluate.
            
        Returns:
            Fitness score in [0, 1].
        """
        if not task.requirements:
            return 1.0
        
        scores = []
        for skill, required in task.requirements.items():
            actual = agent.capabilities.get(skill, 0.0)
            scores.append(min(1.0, actual / (required + 0.01)))
        
        return sum(scores) / len(scores) if scores else 1.0
