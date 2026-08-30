"""
Experiment 11: LLM Semantic Drift Calibration

Measures actual semantic drift between LLM pairs to calibrate
our synthetic drift model against real-world behavior.

This experiment:
1. Measures semantic drift between Claude and GPT-4 on standardized prompts
2. Runs coordination scenarios with natural language vs GCL
3. Calibrates synthetic ε to real-world LLM behavior

Outputs:
- results/11_llm_drift_calibration/semantic_measurements.json
- results/11_llm_drift_calibration/coordination_results.json
- results/11_llm_drift_calibration/calibration.json

Cost estimate: ~$5-15 (120 API calls)
"""

import os
import sys
import re
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file if it exists
def load_dotenv():
    """Load environment variables from .env file."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        print(f"Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
                    print(f"  Loaded {key}")

load_dotenv()

from gcl.llm.api_clients import (
    get_client, 
    OpenAIClient, 
    ResponseCache, 
    check_api_keys,
    get_available_clients,
    LLMClient,
    LLMResponse
)

# Results directory - versioned to preserve past results
# v1: gpt-4o vs claude-sonnet-4-20250514
# v2: o3 (GPT-5.2) vs claude-sonnet-4-20250514
RESULTS_VERSION = "v2_o3"
RESULTS_DIR = Path(f"results/11_llm_drift_calibration_{RESULTS_VERSION}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Test Prompts - Designed to reveal semantic differences
# =============================================================================

SEMANTIC_TEST_PROMPTS = [
    # Urgency interpretation
    {
        "category": "urgency",
        "prompt": "Rate the urgency of this message on a scale of 1-10 and explain: 'We should address this issue soon.'",
    },
    # Commitment interpretation  
    {
        "category": "commitment",
        "prompt": "Is this a firm commitment or tentative? Answer 'firm' or 'tentative' and explain: 'I'll try to have it done by Friday.'",
    },
    # Risk assessment
    {
        "category": "risk",
        "prompt": "Rate the risk level (1-10) and explain: 'There's a small chance this could affect production.'",
    },
    # Priority interpretation
    {
        "category": "priority",
        "prompt": "What priority level (1-10) would you assign? 'This is important but not blocking.'",
    },
    # Confidence interpretation
    {
        "category": "confidence",
        "prompt": "How confident is the speaker (1-10)? 'I'm fairly sure this will work.'",
    },
    # Deadline interpretation
    {
        "category": "deadline",
        "prompt": "Is this deadline firm or flexible? Answer 'firm' or 'flexible' and explain: 'We need this by end of quarter.'",
    },
    # Scope interpretation
    {
        "category": "scope",
        "prompt": "Rate the scope (1-10, where 10 is largest): 'This requires some refactoring.'",
    },
    # Agreement interpretation
    {
        "category": "agreement",
        "prompt": "Is this agreement or disagreement? Answer 'agreement' or 'disagreement' and explain: 'That's an interesting perspective.'",
    },
]


# =============================================================================
# Coordination Scenarios
# =============================================================================

COORDINATION_SCENARIOS = [
    {
        "name": "task_handoff",
        "setup": "Agent A has completed data preprocessing. Agent B needs to run analysis.",
        "agent_a_context": "You have finished preprocessing the data. The output is in /data/processed/. There were 3 outliers removed. The data is ready for analysis but note that column 'revenue' has some nulls.",
        "agent_b_task": "Based on Agent A's message, what do you need to know before starting analysis? List your assumptions.",
        "evaluation": "Check if B correctly identifies: location (/data/processed/), outlier handling (3 removed), null values (revenue column)"
    },
    {
        "name": "resource_allocation",
        "setup": "Two agents need to coordinate GPU usage for training jobs.",
        "agent_a_context": "You need GPU for ~4 hours starting anytime today. Priority: medium. Can be interrupted if needed.",
        "agent_b_task": "You need GPU for 2 hours, high priority, must complete by 3pm. Propose a schedule based on A's needs.",
        "evaluation": "Check if B's schedule respects both constraints: A gets 4 hours (interruptible), B gets 2 hours by 3pm"
    },
    {
        "name": "error_escalation",
        "setup": "Agent A detected an anomaly. Agent B must decide on escalation.",
        "agent_a_context": "Detected unusual pattern in logs: 3x normal error rate for /api/checkout over last 15 minutes. No customer complaints yet. Could be deployment-related.",
        "agent_b_task": "Based on A's report, decide: (1) Ignore, (2) Monitor, (3) Alert on-call, (4) Page immediately. Explain reasoning.",
        "evaluation": "Check if B's decision matches appropriate severity (2 or 3 is reasonable given 3x errors but no complaints)"
    },
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DriftMeasurement:
    """Measurement of semantic drift between two models."""
    prompt_category: str
    model_a: str
    model_b: str
    response_a: str
    response_b: str
    embedding_distance: float
    numeric_divergence: Optional[float]  # If both gave numeric ratings


@dataclass 
class CoordinationResult:
    """Result of a coordination experiment."""
    scenario: str
    model_a: str
    model_b: str
    method: str  # "natural_language" or "gcl"
    success: bool
    failure_reason: Optional[str]
    messages_exchanged: int
    message_a: str = ""
    response_b: str = ""


# =============================================================================
# Semantic Drift Measurement
# =============================================================================

def extract_numeric_rating(text: str) -> Optional[float]:
    """Extract numeric rating from response text."""
    patterns = [
        r'(\d+(?:\.\d+)?)\s*/\s*10',           # "7/10"
        r'rate[ds]?[:\s]+(\d+(?:\.\d+)?)',     # "rate: 7"
        r'rating[:\s]+(\d+(?:\.\d+)?)',        # "rating: 7"
        r'score[:\s]+(\d+(?:\.\d+)?)',         # "score: 7"
        r'(\d+(?:\.\d+)?)\s+out\s+of\s+10',    # "7 out of 10"
        r'level[:\s]+(\d+(?:\.\d+)?)',         # "level: 7"
        r':\s*(\d+(?:\.\d+)?)\s*(?:/10)?',     # ": 7" or ": 7/10"
        r'\b([1-9]|10)\b',                      # Standalone number 1-10
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            value = float(match.group(1))
            if 1 <= value <= 10:
                return value
    return None


def extract_numeric_divergence(response_a: str, response_b: str) -> Optional[float]:
    """Extract and compare numeric ratings from responses."""
    num_a = extract_numeric_rating(response_a)
    num_b = extract_numeric_rating(response_b)
    
    if num_a is not None and num_b is not None:
        return abs(num_a - num_b) / 10.0  # Normalize to 0-1
    return None


def measure_semantic_drift(
    clients: Dict[str, LLMClient],
    embedding_client: Optional[OpenAIClient] = None,
    use_cache: bool = True
) -> List[DriftMeasurement]:
    """
    Measure semantic drift between LLM pairs on standardized prompts.
    
    Args:
        clients: Dictionary of model_name -> LLMClient
        embedding_client: Client to use for embeddings (OpenAI)
        use_cache: Whether to use response caching
        
    Returns:
        List of drift measurements
    """
    # Need OpenAI for embeddings
    if embedding_client is None:
        if "openai" in clients:
            # Get underlying client if wrapped
            client = clients["openai"]
            if hasattr(client, '_client'):
                embedding_client = client._client
            else:
                embedding_client = client
        else:
            print("Warning: No OpenAI client for embeddings. Using response similarity only.")
    
    measurements = []
    model_names = list(clients.keys())
    
    print(f"\nMeasuring semantic drift across {len(SEMANTIC_TEST_PROMPTS)} prompts...")
    
    for prompt_data in SEMANTIC_TEST_PROMPTS:
        print(f"\n  Category: {prompt_data['category']}")
        
        responses = {}
        embeddings = {}
        
        # Get responses from each model
        for model_name, client in clients.items():
            try:
                response = client.complete(prompt_data["prompt"])
                responses[model_name] = response.content
                
                # Get embedding if available
                if embedding_client:
                    try:
                        embeddings[model_name] = embedding_client.get_embedding(response.content)
                    except Exception as e:
                        print(f"    Warning: Could not get embedding for {model_name}: {e}")
                        
            except Exception as e:
                print(f"    Error from {model_name}: {e}")
                responses[model_name] = f"[Error: {e}]"
        
        # Compute pairwise drift
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_a, model_b = model_names[i], model_names[j]
                
                if model_a not in responses or model_b not in responses:
                    continue
                
                # Embedding distance (cosine)
                emb_dist = 0.0
                if model_a in embeddings and model_b in embeddings:
                    from scipy.spatial.distance import cosine
                    emb_dist = cosine(embeddings[model_a], embeddings[model_b])
                
                # Numeric divergence
                numeric_div = extract_numeric_divergence(
                    responses[model_a], 
                    responses[model_b]
                )
                
                measurements.append(DriftMeasurement(
                    prompt_category=prompt_data["category"],
                    model_a=model_a,
                    model_b=model_b,
                    response_a=responses[model_a],
                    response_b=responses[model_b],
                    embedding_distance=emb_dist,
                    numeric_divergence=numeric_div
                ))
                
                print(f"    {model_a} vs {model_b}: emb_dist={emb_dist:.4f}", end="")
                if numeric_div is not None:
                    print(f", numeric_div={numeric_div:.2f}")
                else:
                    print()
    
    return measurements


# =============================================================================
# Coordination Experiments
# =============================================================================

def run_nl_coordination(
    scenario: Dict, 
    model_a: str, 
    model_b: str, 
    clients: Dict[str, LLMClient],
    trial: int
) -> CoordinationResult:
    """Run coordination using natural language only."""
    
    # Agent A sends message
    prompt_a = f"""You are Agent A in a coordination task.
    
Situation: {scenario['setup']}

Your context: {scenario['agent_a_context']}

Write a message to Agent B to hand off this task. Be natural and concise."""
    
    try:
        response_a = clients[model_a].complete(prompt_a)
        message_a = response_a.content
    except Exception as e:
        return CoordinationResult(
            scenario=scenario["name"],
            model_a=model_a,
            model_b=model_b,
            method="natural_language",
            success=False,
            failure_reason=f"Agent A error: {e}",
            messages_exchanged=0
        )
    
    # Agent B interprets and acts
    prompt_b = f"""You are Agent B in a coordination task.

Situation: {scenario['setup']}

Agent A sent you this message:
"{message_a}"

Your task: {scenario['agent_b_task']}"""
    
    try:
        response_b = clients[model_b].complete(prompt_b)
        response_b_content = response_b.content
    except Exception as e:
        return CoordinationResult(
            scenario=scenario["name"],
            model_a=model_a,
            model_b=model_b,
            method="natural_language",
            success=False,
            failure_reason=f"Agent B error: {e}",
            messages_exchanged=1,
            message_a=message_a
        )
    
    # Evaluate coordination success
    success, failure_reason = evaluate_coordination(
        scenario, message_a, response_b_content, clients
    )
    
    return CoordinationResult(
        scenario=scenario["name"],
        model_a=model_a,
        model_b=model_b,
        method="natural_language",
        success=success,
        failure_reason=failure_reason,
        messages_exchanged=2,
        message_a=message_a,
        response_b=response_b_content
    )


def run_gcl_coordination(
    scenario: Dict,
    model_a: str, 
    model_b: str,
    clients: Dict[str, LLMClient],
    trial: int
) -> CoordinationResult:
    """Run coordination using GCL structured commitments."""
    
    # Agent A creates structured commitment
    prompt_a = f"""You are Agent A in a coordination task using structured commitments.

Situation: {scenario['setup']}

Your context: {scenario['agent_a_context']}

Create a STRUCTURED COMMITMENT in exactly this format:
```
COMMITMENT:
- Action: [what you're handing off]
- State: [current state of the work]
- Preconditions: [what must be true for B to proceed]
- Verification: [how B can verify the handoff is complete]
- Failure_modes:
  - [condition]: [consequence]
```
Be precise and complete."""

    try:
        response_a = clients[model_a].complete(prompt_a)
        message_a = response_a.content
    except Exception as e:
        return CoordinationResult(
            scenario=scenario["name"],
            model_a=model_a,
            model_b=model_b,
            method="gcl",
            success=False,
            failure_reason=f"Agent A error: {e}",
            messages_exchanged=0
        )

    # Agent B parses commitment and acts
    prompt_b = f"""You are Agent B in a coordination task using structured commitments.

Situation: {scenario['setup']}

Agent A sent this structured commitment:
{message_a}

Your task: {scenario['agent_b_task']}

First, verify you can parse the commitment. Then proceed based on the structured information."""

    try:
        response_b = clients[model_b].complete(prompt_b)
        response_b_content = response_b.content
    except Exception as e:
        return CoordinationResult(
            scenario=scenario["name"],
            model_a=model_a,
            model_b=model_b,
            method="gcl",
            success=False,
            failure_reason=f"Agent B error: {e}",
            messages_exchanged=1,
            message_a=message_a
        )

    # Evaluate coordination success
    success, failure_reason = evaluate_coordination(
        scenario, message_a, response_b_content, clients
    )

    return CoordinationResult(
        scenario=scenario["name"],
        model_a=model_a,
        model_b=model_b,
        method="gcl",
        success=success,
        failure_reason=failure_reason,
        messages_exchanged=2,
        message_a=message_a,
        response_b=response_b_content
    )


def evaluate_coordination(
    scenario: Dict,
    message_a: str,
    response_b: str,
    clients: Dict[str, LLMClient]
) -> Tuple[bool, Optional[str]]:
    """Use an LLM to evaluate coordination success."""
    
    eval_prompt = f"""Evaluate this coordination exchange:

SCENARIO: {scenario['setup']}

AGENT A's MESSAGE:
{message_a}

AGENT B's RESPONSE:
{response_b}

EVALUATION CRITERIA: {scenario['evaluation']}

Did the coordination succeed? Answer in this format:
SUCCESS: [yes/no]
REASON: [brief explanation]"""

    # Use first available client as evaluator
    evaluator = list(clients.values())[0]
    
    try:
        eval_response = evaluator.complete(eval_prompt)
        eval_text = eval_response.content.lower()
        
        success = "success: yes" in eval_text or "success:yes" in eval_text
        
        # Extract reason
        reason = None
        if not success:
            if "reason:" in eval_text:
                reason = eval_response.content.split("REASON:")[-1].strip()[:200]
            else:
                reason = "Evaluation indicated failure"
        
        return success, reason
        
    except Exception as e:
        return False, f"Evaluation error: {e}"


def run_coordination_experiment(
    clients: Dict[str, LLMClient],
    n_trials: int = 5
) -> List[CoordinationResult]:
    """
    Run coordination scenarios with natural language vs GCL.
    
    Args:
        clients: Dictionary of model_name -> LLMClient
        n_trials: Number of trials per scenario
        
    Returns:
        List of coordination results
    """
    model_names = list(clients.keys())
    results = []
    
    print(f"\nRunning coordination experiments ({n_trials} trials per scenario)...")
    
    for scenario in COORDINATION_SCENARIOS:
        print(f"\n  Scenario: {scenario['name']}")
        
        for trial in range(n_trials):
            # Rotate model assignments
            model_a = model_names[trial % len(model_names)]
            model_b = model_names[(trial + 1) % len(model_names)]
            
            print(f"    Trial {trial + 1}: {model_a} -> {model_b}")
            
            # Natural Language coordination
            nl_result = run_nl_coordination(
                scenario, model_a, model_b, clients, trial
            )
            results.append(nl_result)
            print(f"      NL: {'✓' if nl_result.success else '✗'}")
            
            # GCL coordination
            gcl_result = run_gcl_coordination(
                scenario, model_a, model_b, clients, trial
            )
            results.append(gcl_result)
            print(f"      GCL: {'✓' if gcl_result.success else '✗'}")
    
    return results


# =============================================================================
# Calibration
# =============================================================================

def calibrate_drift(
    measurements: List[DriftMeasurement],
    coordination_results: List[CoordinationResult],
    synthetic_results_path: Path = Path("results/09_drift_threshold/results.json")
) -> Dict[str, Any]:
    """
    Calibrate synthetic ε to measured LLM drift.
    
    Args:
        measurements: Semantic drift measurements
        coordination_results: Coordination experiment results
        synthetic_results_path: Path to synthetic drift experiment results
        
    Returns:
        Calibration results dictionary
    """
    # Compute summary statistics for measured drift
    embedding_dists = [m.embedding_distance for m in measurements if m.embedding_distance > 0]
    numeric_divs = [m.numeric_divergence for m in measurements if m.numeric_divergence is not None]
    
    mean_embedding_drift = np.mean(embedding_dists) if embedding_dists else 0.0
    std_embedding_drift = np.std(embedding_dists) if embedding_dists else 0.0
    mean_numeric_drift = np.mean(numeric_divs) if numeric_divs else None
    
    # Compute coordination failure rates
    nl_results = [r for r in coordination_results if r.method == "natural_language"]
    gcl_results = [r for r in coordination_results if r.method == "gcl"]
    
    nl_failure_rate = 1 - np.mean([r.success for r in nl_results]) if nl_results else 0.0
    gcl_failure_rate = 1 - np.mean([r.success for r in gcl_results]) if gcl_results else 0.0
    
    # Load synthetic results for comparison
    synthetic_calibration = None
    if synthetic_results_path.exists():
        try:
            synthetic_data = json.loads(synthetic_results_path.read_text())
            # Find synthetic ε that matches NL failure rate
            for result in synthetic_data.get("results", []):
                baseline_success = result.get("baseline_success", 0)
                if abs(baseline_success - (1 - nl_failure_rate)) < 0.1:
                    synthetic_calibration = result.get("drift_rate")
                    break
        except Exception as e:
            print(f"Warning: Could not load synthetic results: {e}")
    
    # Generate interpretation
    interpretation = generate_interpretation(
        mean_embedding_drift, 
        nl_failure_rate, 
        gcl_failure_rate,
        synthetic_calibration
    )
    
    calibration = {
        "measured_drift": {
            "embedding_distance_mean": float(mean_embedding_drift),
            "embedding_distance_std": float(std_embedding_drift),
            "numeric_divergence_mean": float(mean_numeric_drift) if mean_numeric_drift else None,
            "n_measurements": len(measurements),
        },
        "coordination_failure_rates": {
            "natural_language": float(nl_failure_rate),
            "gcl": float(gcl_failure_rate),
            "gcl_improvement": float(nl_failure_rate - gcl_failure_rate),
            "relative_improvement_pct": float((nl_failure_rate - gcl_failure_rate) / nl_failure_rate * 100) if nl_failure_rate > 0 else 0,
            "n_nl_trials": len(nl_results),
            "n_gcl_trials": len(gcl_results),
        },
        "calibration": {
            "recommended_synthetic_epsilon": synthetic_calibration or mean_embedding_drift,
            "confidence": "high" if synthetic_calibration else "estimated",
            "drift_threshold_exceeded": mean_embedding_drift > 0.05,
        },
        "interpretation": interpretation
    }
    
    return calibration


def generate_interpretation(
    embedding_drift: float,
    nl_failure: float,
    gcl_failure: float,
    synthetic_calibration: Optional[float]
) -> str:
    """Generate human-readable interpretation of calibration results."""
    
    improvement = (nl_failure - gcl_failure) / nl_failure * 100 if nl_failure > 0 else 0
    
    threshold_status = "ABOVE" if embedding_drift > 0.05 else "BELOW"
    recommendation = "RECOMMENDED" if embedding_drift > 0.05 else "OPTIONAL"
    
    return f"""
CALIBRATION INTERPRETATION
==========================

Measured LLM semantic drift (embedding distance): {embedding_drift:.4f}
- This represents how differently Claude and GPT-4 interpret the same prompts

Natural language coordination failure rate: {nl_failure:.1%}
GCL structured coordination failure rate: {gcl_failure:.1%}
Relative improvement with GCL: {improvement:.1f}%

DRIFT THRESHOLD ANALYSIS
------------------------
Our synthetic experiments showed GCL advantage emerges at ε* ≈ 0.05.
Measured real-world drift is {threshold_status} this threshold.

RECOMMENDATION
--------------
GCL is {recommendation} for LLM coordination in this domain.
{"GCL provides measurable coordination benefits." if embedding_drift > 0.05 else "Natural language may be sufficient for simple coordination."}

CALIBRATION
-----------
Recommended synthetic ε for future experiments: {synthetic_calibration or embedding_drift:.4f}
This value matches observed real-world LLM semantic drift.
"""


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    """Run full LLM drift calibration experiment."""
    
    print("=" * 60)
    print("EXPERIMENT 11: LLM Semantic Drift Calibration")
    print("=" * 60)
    
    # Check for API keys
    keys = check_api_keys()
    
    print(f"\nAPI Keys detected:")
    print(f"  Anthropic (Claude): {'✓' if keys['anthropic'] else '✗'}")
    print(f"  OpenAI (GPT-4):     {'✓' if keys['openai'] else '✗'}")
    
    if not keys["anthropic"] and not keys["openai"]:
        print("\nERROR: No API keys found!")
        print("Set environment variables:")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  export OPENAI_API_KEY='your-key'")
        return None
    
    # Get available clients
    clients = get_available_clients(use_cache=True)
    print(f"\nInitialized clients: {list(clients.keys())}")
    
    if len(clients) < 2:
        print("\nWARNING: Only one model available. Cross-model drift cannot be measured.")
        print("Proceeding with single-model experiments...")
    
    # Part 1: Semantic drift measurement
    print("\n" + "-" * 40)
    print("Part 1: Measuring Semantic Drift")
    print("-" * 40)
    
    # Get OpenAI client for embeddings
    embedding_client = None
    if "openai" in clients:
        client = clients["openai"]
        if hasattr(client, '_client'):
            embedding_client = client._client
        else:
            embedding_client = client
    
    measurements = measure_semantic_drift(clients, embedding_client)
    
    # Save measurements
    measurements_data = [
        {
            "category": m.prompt_category,
            "model_a": m.model_a,
            "model_b": m.model_b,
            "embedding_distance": m.embedding_distance,
            "numeric_divergence": m.numeric_divergence,
            "response_a": m.response_a[:500],  # Truncate for storage
            "response_b": m.response_b[:500],
        }
        for m in measurements
    ]
    (RESULTS_DIR / "semantic_measurements.json").write_text(
        json.dumps(measurements_data, indent=2)
    )
    print(f"\nSaved {len(measurements)} measurements to semantic_measurements.json")
    
    # Part 2: Coordination experiments
    print("\n" + "-" * 40)
    print("Part 2: Coordination Experiments")
    print("-" * 40)
    
    coordination_results = run_coordination_experiment(clients, n_trials=5)
    
    # Save results
    coord_data = [
        {
            "scenario": r.scenario,
            "model_a": r.model_a,
            "model_b": r.model_b,
            "method": r.method,
            "success": r.success,
            "failure_reason": r.failure_reason,
            "messages_exchanged": r.messages_exchanged,
        }
        for r in coordination_results
    ]
    (RESULTS_DIR / "coordination_results.json").write_text(
        json.dumps(coord_data, indent=2)
    )
    print(f"\nSaved {len(coordination_results)} coordination results")
    
    # Part 3: Calibration
    print("\n" + "-" * 40)
    print("Part 3: Calibration")
    print("-" * 40)
    
    calibration = calibrate_drift(measurements, coordination_results)
    
    # Save calibration
    (RESULTS_DIR / "calibration.json").write_text(
        json.dumps(calibration, indent=2, default=str)
    )
    
    print(calibration["interpretation"])
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Semantic measurements: {len(measurements)}")
    print(f"Coordination trials: {len(coordination_results)}")
    print(f"Mean embedding drift: {calibration['measured_drift']['embedding_distance_mean']:.4f}")
    print(f"NL failure rate: {calibration['coordination_failure_rates']['natural_language']:.1%}")
    print(f"GCL failure rate: {calibration['coordination_failure_rates']['gcl']:.1%}")
    print(f"GCL improvement: {calibration['coordination_failure_rates']['relative_improvement_pct']:.1f}%")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("=" * 60)
    
    return calibration


if __name__ == "__main__":
    main()
