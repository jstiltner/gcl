"""
Strong MAS Baselines for comparison experiments.

This module implements established Multi-Agent Systems coordination
mechanisms to provide rigorous baselines for GCL comparison:

1. Contract Net Protocol (CNP) - Classic task allocation
2. FIPA-ACL - Structured agent communication
3. Auction-Based - Market mechanisms
4. MARL - Multi-Agent Reinforcement Learning

These baselines ensure we're not "straw-manning" by comparing
only against natural language LLM coordination.
"""

from gcl.baselines.contract_net import ContractNetProtocol
from gcl.baselines.fipa_acl import FIPAACLProtocol
from gcl.baselines.auction import AuctionProtocol
from gcl.baselines.marl import MARLBaseline
from gcl.baselines.base import MASBaseline, CoordinationTask, CoordinationResult

__all__ = [
    "MASBaseline",
    "CoordinationTask",
    "CoordinationResult",
    "ContractNetProtocol",
    "FIPAACLProtocol",
    "AuctionProtocol",
    "MARLBaseline",
]
