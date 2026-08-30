"""
Auction-based coordination baseline implementation.

Implements various auction mechanisms for multi-agent coordination:
- First-price sealed-bid auction
- Second-price (Vickrey) auction
- Combinatorial auction for task bundles

Auctions provide market-based coordination with well-understood
game-theoretic properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import numpy as np

from gcl.baselines.base import (
    MASBaseline,
    CoordinationTask,
    CoordinationResult,
    AgentCapability,
)


class AuctionType(str, Enum):
    """Types of auctions."""
    FIRST_PRICE = "first_price"
    SECOND_PRICE = "second_price"  # Vickrey
    COMBINATORIAL = "combinatorial"


@dataclass
class Bid:
    """A bid in an auction."""
    bidder_id: str
    task_id: str
    amount: float
    valuation: float  # True valuation (for analysis)


@dataclass
class AuctionResult:
    """Result of a single auction."""
    task_id: str
    winner_id: Optional[str]
    winning_bid: float
    payment: float  # May differ from bid in second-price
    all_bids: List[Bid]


class AuctionProtocol(MASBaseline):
    """
    Auction-based coordination protocol.
    
    Implements market mechanisms for task allocation:
    - Auctioneer announces tasks
    - Bidders submit sealed bids
    - Winner determination and payment
    
    Supports first-price, second-price (Vickrey), and
    combinatorial auctions.
    
    Attributes:
        auction_type: Type of auction mechanism.
        reserve_price: Minimum acceptable bid.
    """
    
    def __init__(
        self,
        n_agents: int,
        seed: int = 42,
        auction_type: AuctionType = AuctionType.SECOND_PRICE,
        reserve_price: float = 0.1,
    ):
        super().__init__(n_agents, seed)
        self.auction_type = auction_type
        self.reserve_price = reserve_price
        self.auction_results: List[AuctionResult] = []
        self.messages_sent = 0
        self.rng = np.random.default_rng(seed)
    
    def coordinate(
        self,
        tasks: List[CoordinationTask],
    ) -> CoordinationResult:
        """
        Coordinate using auction mechanism.
        
        Args:
            tasks: Tasks to auction.
            
        Returns:
            CoordinationResult with assignments.
        """
        self.auction_results = []
        self.messages_sent = 0
        
        assignments: Dict[str, str] = {}
        total_value = 0.0
        total_payment = 0.0
        time_steps = 0
        busy_agents: Set[str] = set()
        
        if self.auction_type == AuctionType.COMBINATORIAL:
            # Run combinatorial auction for all tasks at once
            result = self._run_combinatorial_auction(tasks, busy_agents)
            assignments = result.assignments
            total_value = result.total_value
            time_steps = result.time_steps
            self.messages_sent = result.messages_sent
        else:
            # Run sequential auctions
            for task in tasks:
                # Announce auction
                self.messages_sent += self.n_agents  # Broadcast
                time_steps += 1
                
                # Collect bids
                bids = self._collect_bids(task, busy_agents)
                self.messages_sent += len(bids)
                time_steps += 1
                
                # Determine winner
                auction_result = self._determine_winner(task, bids)
                self.auction_results.append(auction_result)
                
                if auction_result.winner_id:
                    # Announce winner
                    self.messages_sent += self.n_agents
                    time_steps += 1
                    
                    assignments[task.task_id] = auction_result.winner_id
                    busy_agents.add(auction_result.winner_id)
                    total_value += task.value
                    total_payment += auction_result.payment
        
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
                "protocol": "auction",
                "auction_type": self.auction_type.value,
                "total_payment": total_payment,
                "auctions_held": len(self.auction_results),
                "avg_bids_per_auction": (
                    sum(len(r.all_bids) for r in self.auction_results) /
                    len(self.auction_results) if self.auction_results else 0
                ),
            },
        )
    
    def _collect_bids(
        self,
        task: CoordinationTask,
        busy_agents: Set[str],
    ) -> List[Bid]:
        """
        Collect bids from agents.
        
        Agents bid based on their valuation (capability/cost).
        In first-price, they shade bids. In second-price, they bid true value.
        """
        bids = []
        
        for agent in self.agents:
            if agent.agent_id in busy_agents:
                continue
            
            # Compute true valuation
            capability = self.compute_task_fitness(agent, task)
            valuation = task.value * capability - agent.cost
            
            if valuation > self.reserve_price:
                # Compute bid based on auction type
                if self.auction_type == AuctionType.FIRST_PRICE:
                    # Shade bid (bid less than true value)
                    shade_factor = 0.7 + 0.2 * self.rng.random()
                    bid_amount = valuation * shade_factor
                else:
                    # Second-price: bid true value (dominant strategy)
                    bid_amount = valuation
                
                bids.append(Bid(
                    bidder_id=agent.agent_id,
                    task_id=task.task_id,
                    amount=bid_amount,
                    valuation=valuation,
                ))
        
        return bids
    
    def _determine_winner(
        self,
        task: CoordinationTask,
        bids: List[Bid],
    ) -> AuctionResult:
        """
        Determine auction winner and payment.
        
        First-price: winner pays their bid.
        Second-price: winner pays second-highest bid.
        """
        if not bids:
            return AuctionResult(
                task_id=task.task_id,
                winner_id=None,
                winning_bid=0.0,
                payment=0.0,
                all_bids=[],
            )
        
        # Sort by bid amount (descending)
        sorted_bids = sorted(bids, key=lambda b: b.amount, reverse=True)
        winner = sorted_bids[0]
        
        # Determine payment
        if self.auction_type == AuctionType.FIRST_PRICE:
            payment = winner.amount
        else:  # Second-price
            if len(sorted_bids) > 1:
                payment = sorted_bids[1].amount
            else:
                payment = self.reserve_price
        
        return AuctionResult(
            task_id=task.task_id,
            winner_id=winner.bidder_id,
            winning_bid=winner.amount,
            payment=payment,
            all_bids=bids,
        )
    
    def _run_combinatorial_auction(
        self,
        tasks: List[CoordinationTask],
        busy_agents: Set[str],
    ) -> CoordinationResult:
        """
        Run combinatorial auction for task bundles.
        
        Uses greedy approximation for winner determination
        (optimal is NP-hard).
        """
        # Collect bundle bids
        bundle_bids: List[Tuple[str, List[str], float]] = []
        
        for agent in self.agents:
            if agent.agent_id in busy_agents:
                continue
            
            # Generate bids for individual tasks and small bundles
            available_tasks = list(tasks)
            
            # Individual task bids
            for task in available_tasks:
                capability = self.compute_task_fitness(agent, task)
                valuation = task.value * capability - agent.cost
                if valuation > self.reserve_price:
                    bundle_bids.append((
                        agent.agent_id,
                        [task.task_id],
                        valuation,
                    ))
            
            # Two-task bundle bids (with synergy bonus)
            for i, task1 in enumerate(available_tasks):
                for task2 in available_tasks[i+1:]:
                    cap1 = self.compute_task_fitness(agent, task1)
                    cap2 = self.compute_task_fitness(agent, task2)
                    synergy = 1.1  # 10% synergy bonus
                    valuation = (
                        (task1.value * cap1 + task2.value * cap2) * synergy
                        - agent.cost * 1.5  # Reduced cost for bundle
                    )
                    if valuation > self.reserve_price * 2:
                        bundle_bids.append((
                            agent.agent_id,
                            [task1.task_id, task2.task_id],
                            valuation,
                        ))
        
        self.messages_sent = len(bundle_bids) + self.n_agents
        
        # Greedy winner determination
        assignments: Dict[str, str] = {}
        assigned_tasks: Set[str] = set()
        assigned_agents: Set[str] = set()
        total_value = 0.0
        
        # Sort by value density (value / bundle size)
        sorted_bids = sorted(
            bundle_bids,
            key=lambda b: b[2] / len(b[1]),
            reverse=True,
        )
        
        for agent_id, task_ids, value in sorted_bids:
            # Check if agent and tasks are available
            if agent_id in assigned_agents:
                continue
            if any(tid in assigned_tasks for tid in task_ids):
                continue
            
            # Accept this bundle
            for tid in task_ids:
                assignments[tid] = agent_id
                assigned_tasks.add(tid)
            assigned_agents.add(agent_id)
            
            # Add value of assigned tasks
            for task in tasks:
                if task.task_id in task_ids:
                    total_value += task.value
        
        return CoordinationResult(
            success=len(assignments) > 0,
            assignments=assignments,
            messages_sent=self.messages_sent,
            time_steps=3,  # Announce, bid, award
            total_value=total_value,
            efficiency=0.0,  # Calculated by caller
            metadata={"bundle_bids": len(bundle_bids)},
        )
    
    def get_auction_stats(self) -> Dict:
        """Get statistics about auctions."""
        if not self.auction_results:
            return {}
        
        successful = [r for r in self.auction_results if r.winner_id]
        
        return {
            "total_auctions": len(self.auction_results),
            "successful_auctions": len(successful),
            "avg_winning_bid": (
                sum(r.winning_bid for r in successful) / len(successful)
                if successful else 0
            ),
            "avg_payment": (
                sum(r.payment for r in successful) / len(successful)
                if successful else 0
            ),
            "avg_bids_per_auction": (
                sum(len(r.all_bids) for r in self.auction_results) /
                len(self.auction_results)
            ),
        }
