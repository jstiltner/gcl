"""
Contract Net Protocol (CNP) baseline implementation.

The Contract Net Protocol (Smith, 1980) is a classic task allocation
mechanism where:
1. Manager announces task
2. Contractors submit bids
3. Manager awards contract to best bidder
4. Contractor executes and reports

This provides a strong baseline for task allocation scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from gcl.baselines.base import (
    MASBaseline,
    CoordinationTask,
    CoordinationResult,
    AgentCapability,
)


class CNPMessageType(str, Enum):
    """Contract Net Protocol message types."""
    ANNOUNCE = "announce"  # Manager announces task
    BID = "bid"  # Contractor submits bid
    AWARD = "award"  # Manager awards contract
    REJECT = "reject"  # Manager rejects bid
    REPORT = "report"  # Contractor reports completion


@dataclass
class CNPBid:
    """A bid from a contractor."""
    agent_id: str
    task_id: str
    bid_value: float  # Lower is better (cost)
    capability_score: float
    estimated_time: int = 1


@dataclass
class CNPContract:
    """An awarded contract."""
    task_id: str
    contractor_id: str
    bid_value: float
    completed: bool = False


class ContractNetProtocol(MASBaseline):
    """
    Contract Net Protocol implementation.
    
    Implements the classic CNP for task allocation:
    - One manager per task (round-robin or designated)
    - All other agents are potential contractors
    - Bids based on capability and cost
    - Greedy award to lowest bidder meeting requirements
    
    Attributes:
        bid_timeout: Max time to wait for bids.
        min_capability_threshold: Minimum capability to bid.
    """
    
    def __init__(
        self,
        n_agents: int,
        seed: int = 42,
        bid_timeout: int = 10,
        min_capability_threshold: float = 0.3,
    ):
        super().__init__(n_agents, seed)
        self.bid_timeout = bid_timeout
        self.min_capability_threshold = min_capability_threshold
        self.contracts: List[CNPContract] = []
        self.messages_sent = 0
    
    def coordinate(
        self,
        tasks: List[CoordinationTask],
    ) -> CoordinationResult:
        """
        Coordinate using Contract Net Protocol.
        
        Args:
            tasks: Tasks to allocate.
            
        Returns:
            CoordinationResult with assignments.
        """
        self.messages_sent = 0
        self.contracts = []
        assignments: Dict[str, str] = {}
        total_value = 0.0
        time_steps = 0
        
        # Track which agents are busy
        busy_agents: set = set()
        
        for task in tasks:
            # Manager announces task (round-robin manager selection)
            manager_idx = len(self.contracts) % self.n_agents
            manager = self.agents[manager_idx]
            
            # Announce message
            self.messages_sent += 1
            time_steps += 1
            
            # Collect bids from available contractors
            bids = self._collect_bids(task, manager.agent_id, busy_agents)
            
            # Messages for bids
            self.messages_sent += len(bids)
            time_steps += 1
            
            if bids:
                # Award to best bidder (lowest cost meeting requirements)
                best_bid = self._select_best_bid(bids, task)
                
                if best_bid:
                    # Award message
                    self.messages_sent += 1
                    # Reject messages to others
                    self.messages_sent += len(bids) - 1
                    time_steps += 1
                    
                    # Create contract
                    contract = CNPContract(
                        task_id=task.task_id,
                        contractor_id=best_bid.agent_id,
                        bid_value=best_bid.bid_value,
                        completed=True,
                    )
                    self.contracts.append(contract)
                    assignments[task.task_id] = best_bid.agent_id
                    busy_agents.add(best_bid.agent_id)
                    
                    # Report message
                    self.messages_sent += 1
                    time_steps += best_bid.estimated_time
                    
                    total_value += task.value
        
        # Calculate efficiency
        max_value = sum(t.value for t in tasks)
        efficiency = total_value / max_value if max_value > 0 else 0.0
        
        return CoordinationResult(
            success=len(assignments) > 0,
            assignments=assignments,
            messages_sent=self.messages_sent,
            time_steps=time_steps,
            total_value=total_value,
            efficiency=efficiency,
            metadata={
                "protocol": "contract_net",
                "contracts_awarded": len(self.contracts),
                "tasks_unassigned": len(tasks) - len(assignments),
            },
        )
    
    def _collect_bids(
        self,
        task: CoordinationTask,
        manager_id: str,
        busy_agents: set,
    ) -> List[CNPBid]:
        """
        Collect bids from available contractors.
        
        Args:
            task: Task to bid on.
            manager_id: ID of the manager (excluded from bidding).
            busy_agents: Set of busy agent IDs.
            
        Returns:
            List of bids.
        """
        bids = []
        
        for agent in self.agents:
            # Skip manager and busy agents
            if agent.agent_id == manager_id or agent.agent_id in busy_agents:
                continue
            
            # Check capability
            capability_score = self.compute_task_fitness(agent, task)
            
            if capability_score >= self.min_capability_threshold:
                # Compute bid value (cost adjusted by capability)
                bid_value = agent.cost / (capability_score + 0.01)
                
                bids.append(CNPBid(
                    agent_id=agent.agent_id,
                    task_id=task.task_id,
                    bid_value=bid_value,
                    capability_score=capability_score,
                    estimated_time=max(1, int(2 / (capability_score + 0.1))),
                ))
        
        return bids
    
    def _select_best_bid(
        self,
        bids: List[CNPBid],
        task: CoordinationTask,
    ) -> Optional[CNPBid]:
        """
        Select the best bid for a task.
        
        Uses a weighted combination of bid value and capability.
        
        Args:
            bids: List of bids.
            task: Task being bid on.
            
        Returns:
            Best bid or None.
        """
        if not bids:
            return None
        
        # Score = capability / cost (higher is better)
        def bid_score(bid: CNPBid) -> float:
            return bid.capability_score / (bid.bid_value + 0.01)
        
        return max(bids, key=bid_score)
    
    def get_protocol_stats(self) -> Dict:
        """Get statistics about protocol execution."""
        return {
            "total_contracts": len(self.contracts),
            "completed_contracts": sum(1 for c in self.contracts if c.completed),
            "total_messages": self.messages_sent,
            "avg_messages_per_contract": (
                self.messages_sent / len(self.contracts)
                if self.contracts else 0
            ),
        }
