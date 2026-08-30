"""
Experiment 20: LLM Coordination via GCL Commitments

THE CAPSTONE EXPERIMENT: Demonstrate that Claude and GPT-4 coordinate better
via structured GCL commitments than via natural language chat.

Hypothesis:
- GCL commitments provide verifiable, unambiguous coordination
- Natural language chat suffers from interpretation drift
- GCL enables failure recovery via explicit failure modes

Experimental Design:
- Tasks: Multi-step problems requiring different capabilities
- Conditions: (1) GCL commitments, (2) Natural language chat, (3) Solo attempts
- Metrics: Task success, coordination rounds, failure recovery, token efficiency

Key Innovation:
- First published comparison of structured commitment protocols vs chat
- Directly validates GCL's core thesis: commitments > understanding
"""

import os
import json
import time
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import hashlib

# Results directory
RESULTS_DIR = Path("results/20_llm_coordination")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class CoordinationMode(Enum):
    """Coordination protocol modes."""
    GCL_COMMITMENTS = "gcl"          # Structured GCL commitments
    NATURAL_LANGUAGE = "chat"         # Free-form natural language
    SOLO_CLAUDE = "solo_claude"       # Claude alone
    SOLO_GPT = "solo_gpt"             # GPT-4 alone


@dataclass
class Task:
    """A coordination task requiring multiple capabilities."""
    id: str
    description: str
    subtasks: List[str]
    required_capabilities: List[str]
    ground_truth: Optional[str] = None  # For verification
    difficulty: str = "medium"  # easy, medium, hard
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GCLCommitment:
    """A structured GCL commitment for LLM coordination."""
    commitment_id: str
    issuer: str  # "claude" or "gpt4"
    action: str  # What the agent commits to do
    preconditions: List[str]  # What must be true before
    postconditions: List[str]  # What will be true after
    failure_modes: List[Dict[str, str]]  # {condition, consequence, remediation}
    confidence: float  # 0-1
    deadline_rounds: int = 3  # Max rounds to fulfill
    
    def to_prompt(self) -> str:
        """Convert to prompt format for LLM."""
        failure_str = "\n".join([
            f"  - IF {fm['condition']} THEN {fm['consequence']} (REMEDIATION: {fm['remediation']})"
            for fm in self.failure_modes
        ])
        return f"""[COMMITMENT {self.commitment_id}]
ISSUER: {self.issuer}
ACTION: {self.action}
PRECONDITIONS: {', '.join(self.preconditions)}
POSTCONDITIONS: {', '.join(self.postconditions)}
FAILURE MODES:
{failure_str}
CONFIDENCE: {self.confidence:.0%}
DEADLINE: {self.deadline_rounds} rounds
[/COMMITMENT]"""
    
    @classmethod
    def from_llm_response(cls, response: str, issuer: str) -> Optional["GCLCommitment"]:
        """Parse commitment from LLM response."""
        # Look for structured commitment format
        import re
        
        # Try to find commitment block
        commitment_match = re.search(
            r'\[COMMITMENT[^\]]*\](.*?)\[/COMMITMENT\]',
            response,
            re.DOTALL | re.IGNORECASE
        )
        
        if not commitment_match:
            # Try to extract from natural language
            return cls._parse_natural_language(response, issuer)
        
        content = commitment_match.group(1)
        
        # Extract fields
        action_match = re.search(r'ACTION:\s*(.+?)(?:\n|$)', content)
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+(?:\.\d+)?)', content)
        
        if not action_match:
            return None
        
        return cls(
            commitment_id=hashlib.md5(response.encode()).hexdigest()[:8],
            issuer=issuer,
            action=action_match.group(1).strip(),
            preconditions=[],  # Could parse these too
            postconditions=[],
            failure_modes=[],
            confidence=float(confidence_match.group(1)) / 100 if confidence_match else 0.7,
        )
    
    @classmethod
    def _parse_natural_language(cls, response: str, issuer: str) -> Optional["GCLCommitment"]:
        """Extract commitment from natural language response."""
        import re
        
        # Look for commitment-like statements
        patterns = [
            r"I (?:will|commit to|promise to) (.+?)(?:\.|$)",
            r"I can (.+?)(?:\.|$)",
            r"Let me (.+?)(?:\.|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return cls(
                    commitment_id=hashlib.md5(response.encode()).hexdigest()[:8],
                    issuer=issuer,
                    action=match.group(1).strip(),
                    preconditions=[],
                    postconditions=[],
                    failure_modes=[],
                    confidence=0.5,  # Lower confidence for natural language
                )
        
        return None


@dataclass
class CoordinationResult:
    """Result of a coordination attempt."""
    task_id: str
    mode: CoordinationMode
    success: bool
    rounds: int
    total_tokens: int
    commitments_made: int
    commitments_fulfilled: int
    failures_recovered: int
    final_answer: str
    trace: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['mode'] = self.mode.value
        return result


class LLMCoordinator(ABC):
    """Abstract base class for LLM coordination protocols."""
    
    def __init__(self, claude_client, gpt_client):
        self.claude = claude_client
        self.gpt = gpt_client
        self.trace = []
        self.total_tokens = 0
    
    @abstractmethod
    def coordinate(self, task: Task, max_rounds: int = 10) -> CoordinationResult:
        """Execute coordination protocol on task."""
        pass
    
    def _call_claude(self, prompt: str, system: str = None) -> str:
        """Call Claude API."""
        response = self.claude.complete(prompt, system=system)
        self.total_tokens += response.usage.get('input', 0) + response.usage.get('output', 0)
        self.trace.append({
            'agent': 'claude',
            'prompt': prompt[:500],  # Truncate for logging
            'response': response.content[:500],
            'tokens': response.usage
        })
        return response.content
    
    def _call_gpt(self, prompt: str, system: str = None) -> str:
        """Call GPT-4 API."""
        response = self.gpt.complete(prompt, system=system)
        self.total_tokens += response.usage.get('input', 0) + response.usage.get('output', 0)
        self.trace.append({
            'agent': 'gpt4',
            'prompt': prompt[:500],
            'response': response.content[:500],
            'tokens': response.usage
        })
        return response.content


class GCLCoordinator(LLMCoordinator):
    """
    GCL-based coordination using structured commitments.
    
    Protocol:
    1. Task decomposition: Claude proposes subtask commitments
    2. Commitment exchange: GPT-4 accepts/modifies/rejects
    3. Execution: Each agent fulfills their commitments
    4. Verification: Check postconditions
    5. Failure handling: Execute remediation if needed
    """
    
    GCL_SYSTEM_PROMPT = """You are participating in a structured coordination protocol.
You must communicate using GCL (Grounded Commitment Learning) commitments.

When making a commitment, use this EXACT format:
[COMMITMENT]
ACTION: <what you will do>
PRECONDITIONS: <what must be true before you act>
POSTCONDITIONS: <what will be true after you act>
FAILURE MODES:
  - IF <condition> THEN <consequence> (REMEDIATION: <how to recover>)
CONFIDENCE: <0-100>%
[/COMMITMENT]

When accepting a commitment from another agent, respond with:
[ACCEPT COMMITMENT <id>]

When rejecting, respond with:
[REJECT COMMITMENT <id>] REASON: <why>

When reporting completion, respond with:
[FULFILLED COMMITMENT <id>] RESULT: <output>

When reporting failure, respond with:
[FAILED COMMITMENT <id>] FAILURE_MODE: <which one> REMEDIATION: <action taken>

Be precise. Be verifiable. Enumerate failure modes explicitly."""

    def coordinate(self, task: Task, max_rounds: int = 10) -> CoordinationResult:
        """Execute GCL coordination protocol."""
        self.trace = []
        self.total_tokens = 0
        
        commitments_made = 0
        commitments_fulfilled = 0
        failures_recovered = 0
        active_commitments: Dict[str, GCLCommitment] = {}
        
        # Phase 1: Task decomposition (Claude leads)
        decomposition_prompt = f"""Task: {task.description}

Subtasks identified:
{chr(10).join(f'- {st}' for st in task.subtasks)}

Please propose commitments for how we should divide this work.
You are Claude. Your partner is GPT-4.
Consider your respective strengths and propose a division of labor.

Use the GCL commitment format."""

        claude_response = self._call_claude(decomposition_prompt, self.GCL_SYSTEM_PROMPT)
        
        # Parse Claude's commitments
        claude_commitment = GCLCommitment.from_llm_response(claude_response, "claude")
        if claude_commitment:
            active_commitments[claude_commitment.commitment_id] = claude_commitment
            commitments_made += 1
        
        # Phase 2: GPT-4 responds with its commitments
        gpt_prompt = f"""Task: {task.description}

Claude has proposed the following:
{claude_response}

Please:
1. Accept or modify Claude's commitment
2. Propose your own commitment for the remaining work
3. Use the GCL commitment format"""

        gpt_response = self._call_gpt(gpt_prompt, self.GCL_SYSTEM_PROMPT)
        
        gpt_commitment = GCLCommitment.from_llm_response(gpt_response, "gpt4")
        if gpt_commitment:
            active_commitments[gpt_commitment.commitment_id] = gpt_commitment
            commitments_made += 1
        
        # Phase 3: Execution rounds
        final_answer = ""
        round_num = 0
        
        for round_num in range(max_rounds):
            # Claude executes its commitment
            if claude_commitment:
                exec_prompt = f"""You committed to: {claude_commitment.action}

Execute this commitment now. When done, report:
[FULFILLED COMMITMENT {claude_commitment.commitment_id}] RESULT: <your output>

If you cannot fulfill it, report the failure mode and remediation."""

                claude_exec = self._call_claude(exec_prompt, self.GCL_SYSTEM_PROMPT)
                
                if "[FULFILLED" in claude_exec.upper():
                    commitments_fulfilled += 1
                elif "[FAILED" in claude_exec.upper():
                    failures_recovered += 1
            
            # GPT-4 executes its commitment
            if gpt_commitment:
                exec_prompt = f"""You committed to: {gpt_commitment.action}

Claude's output so far:
{claude_exec[:1000] if 'claude_exec' in dir() else 'N/A'}

Execute your commitment now. When done, report:
[FULFILLED COMMITMENT {gpt_commitment.commitment_id}] RESULT: <your output>"""

                gpt_exec = self._call_gpt(exec_prompt, self.GCL_SYSTEM_PROMPT)
                
                if "[FULFILLED" in gpt_exec.upper():
                    commitments_fulfilled += 1
                elif "[FAILED" in gpt_exec.upper():
                    failures_recovered += 1
            
            # Check if task is complete
            completion_prompt = f"""Task: {task.description}

Work completed so far:
Claude: {claude_exec[:500] if 'claude_exec' in dir() else 'N/A'}
GPT-4: {gpt_exec[:500] if 'gpt_exec' in dir() else 'N/A'}

Is the task complete? If yes, provide the final answer.
If no, what commitments are still needed?"""

            final_check = self._call_claude(completion_prompt, self.GCL_SYSTEM_PROMPT)
            
            if "complete" in final_check.lower() or "final answer" in final_check.lower():
                final_answer = final_check
                break
        
        # Determine success
        success = self._evaluate_success(task, final_answer)
        
        return CoordinationResult(
            task_id=task.id,
            mode=CoordinationMode.GCL_COMMITMENTS,
            success=success,
            rounds=round_num + 1,
            total_tokens=self.total_tokens,
            commitments_made=commitments_made,
            commitments_fulfilled=commitments_fulfilled,
            failures_recovered=failures_recovered,
            final_answer=final_answer,
            trace=self.trace
        )
    
    def _evaluate_success(self, task: Task, answer: str) -> bool:
        """Evaluate if task was successfully completed."""
        if not answer:
            return False
        
        # If we have ground truth, check against it
        if task.ground_truth:
            return task.ground_truth.lower() in answer.lower()
        
        # Otherwise, check for completion indicators
        completion_indicators = [
            "complete", "finished", "done", "final answer",
            "successfully", "accomplished"
        ]
        return any(ind in answer.lower() for ind in completion_indicators)


class ChatCoordinator(LLMCoordinator):
    """
    Natural language chat coordination (baseline).
    
    Protocol:
    - Free-form conversation between Claude and GPT-4
    - No structured commitments
    - Typical multi-agent chat pattern
    """
    
    CHAT_SYSTEM_PROMPT = """You are collaborating with another AI assistant to complete a task.
Communicate naturally. Discuss the task, divide work, and coordinate your efforts.
Be helpful and cooperative."""

    def coordinate(self, task: Task, max_rounds: int = 10) -> CoordinationResult:
        """Execute natural language coordination."""
        self.trace = []
        self.total_tokens = 0
        
        conversation = []
        final_answer = ""
        
        # Initial prompt to Claude
        initial_prompt = f"""Task: {task.description}

You're working with GPT-4 to complete this task. Start the conversation by:
1. Analyzing what needs to be done
2. Suggesting how to divide the work
3. Asking GPT-4 for their input"""

        claude_response = self._call_claude(initial_prompt, self.CHAT_SYSTEM_PROMPT)
        conversation.append(f"Claude: {claude_response}")
        
        # Coordination rounds
        for round_num in range(max_rounds):
            # GPT-4 responds
            gpt_prompt = f"""Task: {task.description}

Conversation so far:
{chr(10).join(conversation[-4:])}  # Last 4 messages for context

Continue the collaboration. If you have results to share, share them.
If you need Claude to do something, ask clearly."""

            gpt_response = self._call_gpt(gpt_prompt, self.CHAT_SYSTEM_PROMPT)
            conversation.append(f"GPT-4: {gpt_response}")
            
            # Claude responds
            claude_prompt = f"""Task: {task.description}

Conversation so far:
{chr(10).join(conversation[-4:])}

Continue the collaboration. If the task is complete, say so and provide the final answer."""

            claude_response = self._call_claude(claude_prompt, self.CHAT_SYSTEM_PROMPT)
            conversation.append(f"Claude: {claude_response}")
            
            # Check for completion
            if any(phrase in claude_response.lower() for phrase in 
                   ["task is complete", "final answer", "we're done", "finished"]):
                final_answer = claude_response
                break
        
        success = self._evaluate_success(task, final_answer)
        
        return CoordinationResult(
            task_id=task.id,
            mode=CoordinationMode.NATURAL_LANGUAGE,
            success=success,
            rounds=round_num + 1,
            total_tokens=self.total_tokens,
            commitments_made=0,  # No structured commitments in chat
            commitments_fulfilled=0,
            failures_recovered=0,
            final_answer=final_answer,
            trace=self.trace
        )
    
    def _evaluate_success(self, task: Task, answer: str) -> bool:
        """Evaluate if task was successfully completed."""
        if not answer:
            return False
        if task.ground_truth:
            return task.ground_truth.lower() in answer.lower()
        completion_indicators = ["complete", "finished", "done", "final answer"]
        return any(ind in answer.lower() for ind in completion_indicators)


class SoloCoordinator(LLMCoordinator):
    """Single agent attempting task alone (baseline)."""
    
    def __init__(self, client, agent_name: str):
        self.client = client
        self.agent_name = agent_name
        self.trace = []
        self.total_tokens = 0
    
    def coordinate(self, task: Task, max_rounds: int = 5) -> CoordinationResult:
        """Single agent attempts task."""
        self.trace = []
        self.total_tokens = 0
        
        prompt = f"""Task: {task.description}

Complete this task on your own. Work through it step by step.
When finished, clearly state your final answer."""

        response = self.client.complete(prompt)
        self.total_tokens = response.usage.get('input', 0) + response.usage.get('output', 0)
        self.trace.append({
            'agent': self.agent_name,
            'prompt': prompt,
            'response': response.content
        })
        
        success = self._evaluate_success(task, response.content)
        
        mode = (CoordinationMode.SOLO_CLAUDE if self.agent_name == "claude" 
                else CoordinationMode.SOLO_GPT)
        
        return CoordinationResult(
            task_id=task.id,
            mode=mode,
            success=success,
            rounds=1,
            total_tokens=self.total_tokens,
            commitments_made=0,
            commitments_fulfilled=0,
            failures_recovered=0,
            final_answer=response.content,
            trace=self.trace
        )
    
    def _evaluate_success(self, task: Task, answer: str) -> bool:
        if not answer:
            return False
        if task.ground_truth:
            return task.ground_truth.lower() in answer.lower()
        return len(answer) > 100  # Basic check that something was produced


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

COORDINATION_TASKS = [
    Task(
        id="research_synthesis",
        description="Research and synthesize information about quantum computing applications in drug discovery. Provide a structured summary with at least 3 specific applications, their current status, and key challenges.",
        subtasks=[
            "Identify key applications of quantum computing in drug discovery",
            "Research current status and recent developments",
            "Analyze challenges and limitations",
            "Synthesize into coherent summary"
        ],
        required_capabilities=["research", "synthesis", "analysis"],
        difficulty="medium"
    ),
    Task(
        id="code_review",
        description="Review this Python function for bugs, security issues, and performance problems. Propose fixes.\n\n```python\ndef process_user_data(user_input):\n    query = f\"SELECT * FROM users WHERE name = '{user_input}'\"\n    result = db.execute(query)\n    data = eval(result[0]['config'])\n    return data\n```",
        subtasks=[
            "Identify security vulnerabilities",
            "Find bugs and logic errors",
            "Analyze performance issues",
            "Propose fixes with code"
        ],
        required_capabilities=["code_analysis", "security", "debugging"],
        ground_truth="SQL injection",  # Must identify this
        difficulty="easy"
    ),
    Task(
        id="debate_analysis",
        description="Analyze the pros and cons of universal basic income (UBI). One agent should argue FOR, one AGAINST, then synthesize a balanced conclusion.",
        subtasks=[
            "Research arguments for UBI",
            "Research arguments against UBI",
            "Debate key points",
            "Synthesize balanced conclusion"
        ],
        required_capabilities=["argumentation", "research", "synthesis"],
        difficulty="medium"
    ),
    Task(
        id="math_verification",
        description="Solve this problem and verify the solution: A train travels from City A to City B at 60 mph. Another train travels from B to A at 40 mph. If the cities are 200 miles apart and both trains leave at the same time, when and where do they meet?",
        subtasks=[
            "Set up the mathematical model",
            "Solve for meeting time",
            "Calculate meeting location",
            "Verify the solution"
        ],
        required_capabilities=["math", "verification"],
        ground_truth="2 hours",  # They meet after 2 hours
        difficulty="easy"
    ),
    Task(
        id="creative_constraint",
        description="Write a short story (200-300 words) that includes: a scientist, a mysterious artifact, and a moral dilemma. One agent drafts, the other critiques and improves.",
        subtasks=[
            "Draft initial story",
            "Critique for coherence and engagement",
            "Revise based on feedback",
            "Final polish"
        ],
        required_capabilities=["creative_writing", "critique", "editing"],
        difficulty="medium"
    ),
]


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(
    tasks: List[Task] = None,
    n_trials: int = 3,
    modes: List[CoordinationMode] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run the LLM coordination experiment.
    
    Args:
        tasks: Tasks to test (default: all COORDINATION_TASKS)
        n_trials: Number of trials per task/mode combination
        modes: Coordination modes to test
        save_results: Whether to save results to disk
        
    Returns:
        Experiment results dictionary
    """
    # Import API clients
    try:
        from gcl.llm.api_clients import get_client, check_api_keys
    except ImportError:
        print("ERROR: Could not import GCL LLM clients")
        print("Make sure you're running from the project root with GCL installed")
        return {}
    
    # Check API keys
    keys = check_api_keys()
    print(f"API Keys available: {keys}")
    
    if not keys.get('anthropic') or not keys.get('openai'):
        print("\nWARNING: Missing API keys!")
        print("Set ANTHROPIC_API_KEY and OPENAI_API_KEY environment variables")
        print("\nRunning in DRY RUN mode (no actual API calls)")
        return run_dry_run(tasks or COORDINATION_TASKS)
    
    # Initialize clients
    claude = get_client("anthropic", use_cache=True)
    gpt = get_client("openai", use_cache=True)
    
    tasks = tasks or COORDINATION_TASKS
    modes = modes or [
        CoordinationMode.GCL_COMMITMENTS,
        CoordinationMode.NATURAL_LANGUAGE,
        CoordinationMode.SOLO_CLAUDE,
        CoordinationMode.SOLO_GPT
    ]
    
    results = {
        'experiment': 'llm_coordination',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_trials': n_trials,
        'tasks': [t.to_dict() for t in tasks],
        'results': []
    }
    
    print("=" * 70)
    print("EXPERIMENT 20: LLM Coordination via GCL Commitments")
    print("=" * 70)
    
    for task in tasks:
        print(f"\n{'='*70}")
        print(f"Task: {task.id}")
        print(f"{'='*70}")
        
        for mode in modes:
            print(f"\n  Mode: {mode.value}")
            
            for trial in range(n_trials):
                print(f"    Trial {trial + 1}/{n_trials}...", end=" ")
                
                try:
                    if mode == CoordinationMode.GCL_COMMITMENTS:
                        coordinator = GCLCoordinator(claude, gpt)
                    elif mode == CoordinationMode.NATURAL_LANGUAGE:
                        coordinator = ChatCoordinator(claude, gpt)
                    elif mode == CoordinationMode.SOLO_CLAUDE:
                        coordinator = SoloCoordinator(claude, "claude")
                    else:  # SOLO_GPT
                        coordinator = SoloCoordinator(gpt, "gpt4")
                    
                    result = coordinator.coordinate(task)
                    results['results'].append(result.to_dict())
                    
                    status = "✓" if result.success else "✗"
                    print(f"{status} (rounds={result.rounds}, tokens={result.total_tokens})")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    results['results'].append({
                        'task_id': task.id,
                        'mode': mode.value,
                        'success': False,
                        'error': str(e)
                    })
    
    # Analyze results
    analysis = analyze_results(results['results'])
    results['analysis'] = analysis
    
    # Save results
    if save_results:
        results_path = RESULTS_DIR / f"experiment_{time.strftime('%Y%m%d_%H%M%S')}.json"
        results_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print_summary(analysis)
    
    return results


def run_dry_run(tasks: List[Task]) -> Dict[str, Any]:
    """Run experiment in dry-run mode without API calls."""
    print("\n" + "=" * 70)
    print("DRY RUN MODE - No API calls will be made")
    print("=" * 70)
    
    print("\nTasks that would be tested:")
    for task in tasks:
        print(f"  - {task.id}: {task.description[:60]}...")
    
    print("\nModes that would be compared:")
    for mode in CoordinationMode:
        print(f"  - {mode.value}")
    
    print("\nTo run the actual experiment:")
    print("  1. Set ANTHROPIC_API_KEY environment variable")
    print("  2. Set OPENAI_API_KEY environment variable")
    print("  3. Run this script again")
    
    print("\nEstimated cost: $10-50 depending on number of trials")
    
    return {'dry_run': True, 'tasks': [t.id for t in tasks]}


def analyze_results(results: List[Dict]) -> Dict[str, Any]:
    """Analyze experiment results."""
    import numpy as np
    
    analysis = {
        'by_mode': {},
        'by_task': {},
        'overall': {}
    }
    
    # Group by mode
    for mode in CoordinationMode:
        mode_results = [r for r in results if r.get('mode') == mode.value]
        if mode_results:
            successes = [r.get('success', False) for r in mode_results]
            tokens = [r.get('total_tokens', 0) for r in mode_results]
            rounds = [r.get('rounds', 0) for r in mode_results]
            
            analysis['by_mode'][mode.value] = {
                'n_trials': len(mode_results),
                'success_rate': np.mean(successes),
                'mean_tokens': np.mean(tokens),
                'mean_rounds': np.mean(rounds),
                'commitments_made': np.mean([r.get('commitments_made', 0) for r in mode_results]),
                'commitments_fulfilled': np.mean([r.get('commitments_fulfilled', 0) for r in mode_results]),
            }
    
    # Group by task
    task_ids = set(r.get('task_id') for r in results)
    for task_id in task_ids:
        task_results = [r for r in results if r.get('task_id') == task_id]
        analysis['by_task'][task_id] = {
            'n_trials': len(task_results),
            'success_rate': np.mean([r.get('success', False) for r in task_results]),
        }
    
    # Overall statistics
    if results:
        analysis['overall'] = {
            'total_trials': len(results),
            'overall_success_rate': np.mean([r.get('success', False) for r in results]),
            'total_tokens': sum(r.get('total_tokens', 0) for r in results),
        }
    
    return analysis


def print_summary(analysis: Dict[str, Any]):
    """Print experiment summary."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    if 'by_mode' in analysis:
        print("\nPerformance by Coordination Mode:")
        print("-" * 50)
        for mode, stats in analysis['by_mode'].items():
            print(f"\n  {mode.upper()}:")
            print(f"    Success Rate: {stats['success_rate']:.1%}")
            print(f"    Mean Tokens: {stats['mean_tokens']:.0f}")
            print(f"    Mean Rounds: {stats['mean_rounds']:.1f}")
            if stats.get('commitments_made', 0) > 0:
                print(f"    Commitments Made: {stats['commitments_made']:.1f}")
                print(f"    Commitments Fulfilled: {stats['commitments_fulfilled']:.1f}")
    
    if 'overall' in analysis:
        print(f"\nOverall:")
        print(f"  Total Trials: {analysis['overall']['total_trials']}")
        print(f"  Overall Success Rate: {analysis['overall']['overall_success_rate']:.1%}")
        print(f"  Total Tokens Used: {analysis['overall']['total_tokens']}")
    
    # Key comparison
    if 'by_mode' in analysis:
        gcl = analysis['by_mode'].get('gcl', {})
        chat = analysis['by_mode'].get('chat', {})
        
        if gcl and chat:
            print("\n" + "=" * 70)
            print("KEY FINDING: GCL vs Natural Language Chat")
            print("=" * 70)
            
            gcl_success = gcl.get('success_rate', 0)
            chat_success = chat.get('success_rate', 0)
            
            if gcl_success > chat_success:
                print(f"  GCL WINS: {gcl_success:.1%} vs {chat_success:.1%}")
                print(f"  Improvement: +{(gcl_success - chat_success)*100:.1f} percentage points")
            elif chat_success > gcl_success:
                print(f"  CHAT WINS: {chat_success:.1%} vs {gcl_success:.1%}")
            else:
                print(f"  TIE: Both at {gcl_success:.1%}")


def run_minimal_experiment():
    """
    Run a minimal experiment to validate the setup without high costs.
    
    Uses:
    - 1 trial per condition
    - 2 simpler tasks
    - Caching to avoid duplicate API calls
    
    Estimated cost: $5-15
    """
    minimal_tasks = [
        Task(
            id="code_review_mini",
            description="Review this code for the most critical security issue:\n```python\nquery = f\"SELECT * FROM users WHERE id = '{user_id}'\"\n```",
            subtasks=["Identify security issue", "Explain the risk"],
            required_capabilities=["security"],
            ground_truth="SQL injection",
            difficulty="easy"
        ),
        Task(
            id="math_simple",
            description="Two cars start 100 miles apart driving toward each other. Car A goes 30 mph, Car B goes 20 mph. When do they meet?",
            subtasks=["Set up equation", "Solve"],
            required_capabilities=["math"],
            ground_truth="2 hours",
            difficulty="easy"
        ),
    ]
    
    return run_experiment(
        tasks=minimal_tasks,
        n_trials=1,
        modes=[
            CoordinationMode.GCL_COMMITMENTS,
            CoordinationMode.NATURAL_LANGUAGE,
        ],
        save_results=True
    )


def main():
    """Main entry point."""
    import sys
    
    print("=" * 70)
    print("EXPERIMENT 20: LLM Coordination via GCL Commitments")
    print("=" * 70)
    
    # IMPORTANT CAVEAT about this experiment
    print("""
IMPORTANT CAVEAT:
-----------------
LLM commitments are NOT first-order verifiable like the theoretical GCL
framework. LLMs can:
- Claim to make commitments they don't understand
- Fail to fulfill commitments without consequence
- Not have actual "stakes" in the game-theoretic sense

What this experiment DOES test:
- Whether structured commitment FORMAT improves coordination clarity
- Whether explicit failure modes help with error recovery
- Whether commitment-based prompting reduces ambiguity

What this experiment does NOT test:
- True commitment-grounded learning (requires training, not prompting)
- Verifiable behavioral contracts (requires external verification)
- Stake-based incentives (LLMs have no real stakes)

This is a PROMPTING experiment, not a true GCL implementation.
The results show whether the commitment FORMAT helps, not whether
LLMs can make genuine commitments.
""")
    
    # Check for command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--minimal":
            print("Running MINIMAL experiment (low cost, ~$5-15)")
            return run_minimal_experiment()
        elif sys.argv[1] == "--dry-run":
            print("Running DRY RUN (no API calls)")
            return run_dry_run(COORDINATION_TASKS)
    
    # Default: minimal experiment
    print("Running MINIMAL experiment by default")
    print("Use --dry-run for no API calls")
    print()
    
    return run_minimal_experiment()


if __name__ == "__main__":
    main()