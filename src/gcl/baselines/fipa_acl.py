"""
FIPA-ACL Protocol baseline implementation.

FIPA-ACL (Foundation for Intelligent Physical Agents - Agent
Communication Language) provides structured performatives for
agent communication:
- INFORM, REQUEST, PROPOSE, ACCEPT, REJECT, etc.

This baseline implements FIPA-compliant negotiation for
multi-agent coordination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import numpy as np

from gcl.baselines.base import (
    MASBaseline,
    CoordinationTask,
    CoordinationResult,
    AgentCapability,
)


class Performative(str, Enum):
    """FIPA-ACL performatives."""
    # Informative
    INFORM = "inform"
    INFORM_IF = "inform-if"
    INFORM_REF = "inform-ref"
    CONFIRM = "confirm"
    DISCONFIRM = "disconfirm"
    
    # Requesting
    REQUEST = "request"
    REQUEST_WHEN = "request-when"
    REQUEST_WHENEVER = "request-whenever"
    QUERY_IF = "query-if"
    QUERY_REF = "query-ref"
    
    # Negotiating
    PROPOSE = "propose"
    ACCEPT_PROPOSAL = "accept-proposal"
    REJECT_PROPOSAL = "reject-proposal"
    CFP = "cfp"  # Call for proposals
    
    # Action
    AGREE = "agree"
    REFUSE = "refuse"
    CANCEL = "cancel"
    FAILURE = "failure"
    NOT_UNDERSTOOD = "not-understood"


@dataclass
class ACLMessage:
    """
    FIPA-ACL message structure.
    
    Attributes:
        performative: The speech act type.
        sender: Agent ID of sender.
        receiver: Agent ID of receiver.
        content: Message content.
        reply_to: Message ID this replies to.
        conversation_id: ID of the conversation.
        protocol: Protocol being used.
    """
    performative: Performative
    sender: str
    receiver: str
    content: Any
    reply_to: Optional[str] = None
    conversation_id: str = ""
    protocol: str = "fipa-contract-net"
    message_id: str = field(default_factory=lambda: str(np.random.randint(0, 1000000)))


@dataclass
class Proposal:
    """A proposal in FIPA negotiation."""
    agent_id: str
    task_id: str
    offer: Dict[str, Any]
    utility: float


class FIPAACLProtocol(MASBaseline):
    """
    FIPA-ACL based coordination protocol.
    
    Implements structured negotiation using FIPA performatives:
    1. Initiator sends CFP (Call for Proposals)
    2. Participants respond with PROPOSE or REFUSE
    3. Initiator sends ACCEPT-PROPOSAL or REJECT-PROPOSAL
    4. Accepted participant sends INFORM (done) or FAILURE
    
    This provides more structured communication than CNP with
    explicit conversation tracking and protocol compliance.
    """
    
    def __init__(
        self,
        n_agents: int,
        seed: int = 42,
        max_negotiation_rounds: int = 5,
        utility_threshold: float = 0.4,
    ):
        super().__init__(n_agents, seed)
        self.max_negotiation_rounds = max_negotiation_rounds
        self.utility_threshold = utility_threshold
        self.messages: List[ACLMessage] = []
        self.conversations: Dict[str, List[ACLMessage]] = {}
        self.messages_sent = 0
    
    def coordinate(
        self,
        tasks: List[CoordinationTask],
    ) -> CoordinationResult:
        """
        Coordinate using FIPA-ACL protocol.
        
        Args:
            tasks: Tasks to coordinate.
            
        Returns:
            CoordinationResult with assignments.
        """
        self.messages = []
        self.conversations = {}
        self.messages_sent = 0
        
        assignments: Dict[str, str] = {}
        total_value = 0.0
        time_steps = 0
        busy_agents: Set[str] = set()
        
        for task in tasks:
            # Select initiator (round-robin)
            initiator_idx = len(assignments) % self.n_agents
            initiator = self.agents[initiator_idx]
            
            conversation_id = f"conv_{task.task_id}"
            
            # Phase 1: CFP
            cfp_msg = self._send_cfp(initiator, task, conversation_id)
            time_steps += 1
            
            # Phase 2: Collect proposals
            proposals = self._collect_proposals(
                task, initiator.agent_id, conversation_id, busy_agents
            )
            time_steps += 1
            
            if proposals:
                # Phase 3: Evaluate and accept best
                best_proposal = self._evaluate_proposals(proposals, task)
                
                if best_proposal:
                    # Accept best, reject others
                    self._send_accept(initiator, best_proposal, conversation_id)
                    for p in proposals:
                        if p.agent_id != best_proposal.agent_id:
                            self._send_reject(initiator, p, conversation_id)
                    time_steps += 1
                    
                    # Phase 4: Execution confirmation
                    self._send_inform_done(best_proposal.agent_id, task, conversation_id)
                    time_steps += 1
                    
                    assignments[task.task_id] = best_proposal.agent_id
                    busy_agents.add(best_proposal.agent_id)
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
                "protocol": "fipa_acl",
                "conversations": len(self.conversations),
                "total_messages": len(self.messages),
                "performative_counts": self._count_performatives(),
            },
        )
    
    def _send_cfp(
        self,
        initiator: AgentCapability,
        task: CoordinationTask,
        conversation_id: str,
    ) -> ACLMessage:
        """Send Call for Proposals."""
        # Broadcast CFP to all other agents
        for agent in self.agents:
            if agent.agent_id != initiator.agent_id:
                msg = ACLMessage(
                    performative=Performative.CFP,
                    sender=initiator.agent_id,
                    receiver=agent.agent_id,
                    content={
                        "task_id": task.task_id,
                        "requirements": task.requirements,
                        "value": task.value,
                    },
                    conversation_id=conversation_id,
                )
                self._record_message(msg)
        
        return msg
    
    def _collect_proposals(
        self,
        task: CoordinationTask,
        initiator_id: str,
        conversation_id: str,
        busy_agents: Set[str],
    ) -> List[Proposal]:
        """Collect proposals from participants."""
        proposals = []
        
        for agent in self.agents:
            if agent.agent_id == initiator_id or agent.agent_id in busy_agents:
                continue
            
            # Compute utility for this agent
            capability = self.compute_task_fitness(agent, task)
            utility = capability / (agent.cost + 0.01)
            
            if utility >= self.utility_threshold:
                # Send PROPOSE
                msg = ACLMessage(
                    performative=Performative.PROPOSE,
                    sender=agent.agent_id,
                    receiver=initiator_id,
                    content={
                        "task_id": task.task_id,
                        "cost": agent.cost,
                        "capability": capability,
                    },
                    conversation_id=conversation_id,
                )
                self._record_message(msg)
                
                proposals.append(Proposal(
                    agent_id=agent.agent_id,
                    task_id=task.task_id,
                    offer={"cost": agent.cost, "capability": capability},
                    utility=utility,
                ))
            else:
                # Send REFUSE
                msg = ACLMessage(
                    performative=Performative.REFUSE,
                    sender=agent.agent_id,
                    receiver=initiator_id,
                    content={"task_id": task.task_id, "reason": "low_utility"},
                    conversation_id=conversation_id,
                )
                self._record_message(msg)
        
        return proposals
    
    def _evaluate_proposals(
        self,
        proposals: List[Proposal],
        task: CoordinationTask,
    ) -> Optional[Proposal]:
        """Evaluate proposals and select best."""
        if not proposals:
            return None
        
        # Select highest utility proposal
        return max(proposals, key=lambda p: p.utility)
    
    def _send_accept(
        self,
        initiator: AgentCapability,
        proposal: Proposal,
        conversation_id: str,
    ) -> None:
        """Send ACCEPT-PROPOSAL."""
        msg = ACLMessage(
            performative=Performative.ACCEPT_PROPOSAL,
            sender=initiator.agent_id,
            receiver=proposal.agent_id,
            content={"task_id": proposal.task_id},
            conversation_id=conversation_id,
        )
        self._record_message(msg)
    
    def _send_reject(
        self,
        initiator: AgentCapability,
        proposal: Proposal,
        conversation_id: str,
    ) -> None:
        """Send REJECT-PROPOSAL."""
        msg = ACLMessage(
            performative=Performative.REJECT_PROPOSAL,
            sender=initiator.agent_id,
            receiver=proposal.agent_id,
            content={"task_id": proposal.task_id},
            conversation_id=conversation_id,
        )
        self._record_message(msg)
    
    def _send_inform_done(
        self,
        agent_id: str,
        task: CoordinationTask,
        conversation_id: str,
    ) -> None:
        """Send INFORM (task completed)."""
        msg = ACLMessage(
            performative=Performative.INFORM,
            sender=agent_id,
            receiver="initiator",  # Broadcast completion
            content={"task_id": task.task_id, "status": "completed"},
            conversation_id=conversation_id,
        )
        self._record_message(msg)
    
    def _record_message(self, msg: ACLMessage) -> None:
        """Record a message."""
        self.messages.append(msg)
        self.messages_sent += 1
        
        if msg.conversation_id not in self.conversations:
            self.conversations[msg.conversation_id] = []
        self.conversations[msg.conversation_id].append(msg)
    
    def _count_performatives(self) -> Dict[str, int]:
        """Count messages by performative type."""
        counts: Dict[str, int] = {}
        for msg in self.messages:
            perf = msg.performative.value
            counts[perf] = counts.get(perf, 0) + 1
        return counts
    
    def get_conversation(self, conversation_id: str) -> List[ACLMessage]:
        """Get all messages in a conversation."""
        return self.conversations.get(conversation_id, [])
