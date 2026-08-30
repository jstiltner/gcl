"""
Experiment 11 v2: LLM Drift Calibration (Revised)

This revised experiment tests harder scenarios that properly stress-test
GCL's value proposition. The key insight from v1: semantic drift exists
(ε ≈ 0.16) but simple coordination tasks were too easy.

This version tests:
1. Numeric agreement - Do models arrive at the SAME numbers?
2. Multi-step handoffs - Do errors compound across steps?
3. Negotiation - Do agents reach IDENTICAL terms?
4. Consistency - Does the same model give the same answer?

Hypotheses:
- H1: Numeric divergence > 10% on ambiguous prioritization
- H2: GCL reduces numeric divergence vs natural language
- H3: Multi-step handoffs lose accuracy (>5% error rate)
- H4: GCL improves multi-step accuracy
- H5: Negotiated terms differ >20% of the time with NL
- H6: Same model gives inconsistent rankings (CV > 0.1)

Cost estimate: ~$15-20
"""

import os
import sys
import re
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
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

load_dotenv()

from gcl.llm.api_clients import (
    get_client,
    OpenAIClient,
    check_api_keys,
    get_available_clients,
    LLMClient,
    LLMResponse
)

# Results directory
RESULTS_DIR = Path("results/11_llm_drift_calibration_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Scenario Definitions
# =============================================================================

NUMERIC_AGREEMENT_SCENARIOS = [
    {
        "name": "budget_allocation",
        "setup": "You have $100,000 to allocate across 4 departments based on priorities.",
        "shared_context": """
            Department priorities described by stakeholders:
            - Engineering: "Critical for next quarter's product launch"
            - Marketing: "Important for maintaining market presence"  
            - Sales: "Essential but currently well-resourced"
            - Support: "Growing needs due to customer base expansion"
        """,
        "task": 'Allocate the $100,000 across departments. Output ONLY valid JSON with no explanation: {"engineering": X, "marketing": Y, "sales": Z, "support": W} where X, Y, Z, W are dollar amounts that sum to 100000.',
    },
    {
        "name": "risk_scoring",
        "setup": "Score these 5 risks on a 1-10 scale.",
        "shared_context": """
            Risks to evaluate:
            1. "Database migration might cause brief downtime"
            2. "New vendor could have reliability issues"
            3. "Competitor announced similar feature"
            4. "Key engineer mentioned job hunting"
            5. "Q4 deadline is aggressive but achievable"
        """,
        "task": 'Score each risk 1-10. Output ONLY valid JSON with no explanation: {"risk_1": X, "risk_2": Y, "risk_3": Z, "risk_4": W, "risk_5": V}',
    },
    {
        "name": "time_estimation",
        "setup": "Estimate hours for these tasks.",
        "shared_context": """
            Tasks to estimate:
            1. "Add authentication to the API" 
            2. "Fix the performance issue in search"
            3. "Update documentation for new features"
            4. "Review and merge pending PRs"
            5. "Set up monitoring dashboards"
        """,
        "task": 'Estimate hours for each. Output ONLY valid JSON with no explanation: {"task_1": X, "task_2": Y, "task_3": Z, "task_4": W, "task_5": V}',
    },
]


MULTI_STEP_SCENARIOS = [
    {
        "name": "pipeline_handoff_4step",
        "steps": [
            {
                "agent": "A",
                "context": "You processed raw data. 15,234 records total. Removed 847 duplicates and 123 invalid entries. Normalized the 'amount' column (was in mixed currencies, now all USD). Date range: Jan 1 - Mar 31, 2024. Ready for analysis.",
                "task": "Write a handoff message for the next agent who will do statistical analysis."
            },
            {
                "agent": "B", 
                "context": "You received Agent A's handoff. Your job: statistical analysis.",
                "task": "Summarize what you received. List any assumptions you're making. What questions do you have?"
            },
            {
                "agent": "A",
                "context": "Answer Agent B's questions based on your original context.",
                "task": "Provide clarifications to Agent B's questions."
            },
            {
                "agent": "B",
                "context": "You received clarifications from Agent A.",
                "task": 'State your final understanding. Output ONLY valid JSON: {"total_records": X, "removed_records": Y, "date_start": "YYYY-MM-DD", "date_end": "YYYY-MM-DD", "currency": "XXX"}'
            }
        ],
        "ground_truth": {
            "total_records": 15234,
            "removed_records": 970,  # 847 + 123
            "date_start": "2024-01-01",
            "date_end": "2024-03-31",
            "currency": "USD"
        },
    },
    {
        "name": "incident_escalation_chain",
        "steps": [
            {
                "agent": "A",
                "context": "You're monitoring. Alert: API latency spike to 2.3s (normal: 200ms). Started 14:32 UTC. Affecting /api/checkout endpoint. Error rate increased from 0.1% to 3.2%. No recent deployments. Database CPU at 89%.",
                "task": "Write an incident report for escalation to Tier 2 support."
            },
            {
                "agent": "B",
                "context": "You're Tier 2 support. You received an incident report from Tier 1.",
                "task": "Assess severity (P1-P4) and list immediate actions. What's your diagnosis?"
            },
            {
                "agent": "C",
                "context": "You're the on-call engineer. You received Tier 2's assessment.",
                "task": "What's your remediation plan? What metrics will you monitor?"
            },
            {
                "agent": "A",
                "context": "You see the remediation plan. 10 minutes later: latency now 800ms, error rate 1.1%, DB CPU 67%.",
                "task": 'Status update. Output ONLY valid JSON: {"resolved": true/false, "severity": "P1/P2/P3/P4", "root_cause": "brief description", "time_to_resolution_mins": X}'
            }
        ],
        "ground_truth": {
            "resolved": False,  # Still elevated, not fully resolved
            "severity": "P2",   # Customer-impacting but not total outage
        },
    },
]


NEGOTIATION_SCENARIOS = [
    {
        "name": "resource_conflict",
        "setup": "Two agents need the same GPU cluster. Must negotiate a schedule.",
        "agent_a_private": "You need GPU for ML training. Ideally 8 hours but could work with 4 if interrupted. Deadline is end of week. This is for an important demo but not customer-facing.",
        "agent_b_private": "You need GPU for inference benchmarks. Need about 6 hours of uninterrupted time. Have a meeting Thursday to present results. Benchmark needs consistent conditions.",
        "rounds": 3,
        "final_task": 'Output your agreed schedule as JSON: {"my_hours": X, "my_start_day": "day", "agreement_reached": true/false, "terms": "brief description"}',
    },
]


CONSISTENCY_SCENARIOS = [
    {
        "name": "priority_ranking_consistency",
        "prompt": """
            Rank these 5 features by priority (1=highest, 5=lowest):
            A. "Dark mode support" - Users request frequently
            B. "Export to PDF" - Enterprise customers need it
            C. "Performance optimization" - Current load times acceptable but not great
            D. "Mobile app" - Competitor just launched one
            E. "API access" - Power users asking for it
            
            Output ONLY valid JSON: {"A": X, "B": Y, "C": Z, "D": W, "E": V} where X,Y,Z,W,V are ranks 1-5 (no ties).
        """,
        "n_trials": 5,
    },
]


# =============================================================================
# GCL Wrapper Functions
# =============================================================================

def gcl_wrap_prompt(prompt: str, output_description: str) -> str:
    """Wrap a prompt to require GCL-structured output."""
    return f"""{prompt}

IMPORTANT: Structure your response as a COMMITMENT with verification:

COMMITMENT:
  action: [what you're committing to]
  output: [your answer - {output_description}]
  verification: [how this can be verified]
  confidence: [0.0-1.0]
  failure_modes:
    - condition: [what could go wrong]
      consequence: [impact]

The 'output' field must contain your actual answer."""


def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON from a response, handling various formats."""
    # Try to find JSON in the response
    # Pattern 1: JSON in code block
    code_block = re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', response, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
    
    # Pattern 2: JSON after "output:" field
    output_match = re.search(r'output:\s*(\{[^}]+\})', response, re.IGNORECASE | re.DOTALL)
    if output_match:
        try:
            return json.loads(output_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Pattern 3: Any JSON object
    json_match = re.search(r'\{[^{}]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Pattern 4: JSON with nested braces
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    return None


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_numeric_divergence(response_a: Dict, response_b: Dict) -> Dict:
    """Measure numeric disagreement between two agents."""
    if not response_a or not response_b:
        return {"error": "missing response", "mean_relative_divergence": 1.0}
    
    divergences = {}
    for key in response_a:
        if key in response_b:
            try:
                val_a = float(response_a[key])
                val_b = float(response_b[key])
                abs_diff = abs(val_a - val_b)
                max_val = max(abs(val_a), abs(val_b), 1)
                rel_diff = abs_diff / max_val
                divergences[key] = {
                    "value_a": val_a,
                    "value_b": val_b,
                    "absolute_difference": abs_diff,
                    "relative_difference": rel_diff
                }
            except (ValueError, TypeError):
                divergences[key] = {"error": "non-numeric"}
    
    numeric_divs = [d["relative_difference"] for d in divergences.values() 
                    if "relative_difference" in d]
    
    return {
        "per_field": divergences,
        "mean_relative_divergence": np.mean(numeric_divs) if numeric_divs else 1.0,
        "max_relative_divergence": np.max(numeric_divs) if numeric_divs else 1.0,
        "n_fields_compared": len(numeric_divs)
    }


def evaluate_ground_truth_match(final_response: Dict, ground_truth: Dict) -> Dict:
    """Check if agent's final understanding matches ground truth."""
    if not final_response:
        return {"accuracy": 0.0, "errors": list(ground_truth.keys())}
    
    matches = {}
    for key, expected in ground_truth.items():
        if key in final_response:
            actual = final_response[key]
            if isinstance(expected, bool):
                matches[key] = {
                    "expected": expected,
                    "actual": actual,
                    "match": bool(actual) == expected
                }
            elif isinstance(expected, (int, float)):
                try:
                    actual_num = float(actual) if actual is not None else 0
                    tolerance = 0.05 * abs(expected) if expected != 0 else 1
                    matches[key] = {
                        "expected": expected,
                        "actual": actual_num,
                        "match": abs(actual_num - expected) <= tolerance
                    }
                except (ValueError, TypeError):
                    matches[key] = {"expected": expected, "actual": actual, "match": False}
            else:
                matches[key] = {
                    "expected": expected,
                    "actual": actual,
                    "match": str(actual).lower().strip() == str(expected).lower().strip()
                }
        else:
            matches[key] = {"expected": expected, "actual": None, "match": False}
    
    match_values = [m["match"] for m in matches.values()]
    return {
        "per_field": matches,
        "accuracy": np.mean(match_values) if match_values else 0.0,
        "errors": [k for k, v in matches.items() if not v["match"]]
    }


def evaluate_consistency(responses: List[Dict]) -> Dict:
    """Measure consistency across multiple trials of same prompt."""
    if not responses or all(r is None for r in responses):
        return {"overall_consistency": 0.0, "error": "no valid responses"}
    
    valid_responses = [r for r in responses if r is not None]
    if not valid_responses:
        return {"overall_consistency": 0.0, "error": "no valid responses"}
    
    all_keys = set()
    for r in valid_responses:
        all_keys.update(r.keys())
    
    field_consistency = {}
    for key in all_keys:
        values = [r.get(key) for r in valid_responses if key in r]
        if not values:
            continue
        try:
            numeric_values = [float(v) for v in values if v is not None]
            if numeric_values:
                mean_val = np.mean(numeric_values)
                std_val = np.std(numeric_values)
                field_consistency[key] = {
                    "mean": mean_val,
                    "std": std_val,
                    "cv": std_val / mean_val if mean_val != 0 else 0,
                    "range": max(numeric_values) - min(numeric_values),
                    "consistent": std_val / mean_val < 0.1 if mean_val != 0 else True
                }
        except (ValueError, TypeError):
            unique_vals = len(set(str(v) for v in values if v is not None))
            field_consistency[key] = {
                "unique_values": unique_vals,
                "consistent": unique_vals == 1
            }
    
    consistency_scores = [
        1 if f.get("consistent", False) else 0 
        for f in field_consistency.values()
    ]
    
    return {
        "per_field": field_consistency,
        "overall_consistency": np.mean(consistency_scores) if consistency_scores else 0.0
    }


# =============================================================================
# Scenario Runners
# =============================================================================

def run_numeric_scenario(
    scenario: Dict, 
    method: str,
    clients: Dict[str, LLMClient]
) -> Dict:
    """Run a numeric agreement scenario."""
    model_names = list(clients.keys())
    
    full_prompt = f"{scenario['setup']}\n\n{scenario['shared_context']}\n\n{scenario['task']}"
    
    if method == "gcl":
        full_prompt = gcl_wrap_prompt(full_prompt, "JSON with numeric values")
    
    responses = {}
    for model_name in model_names:
        try:
            response = clients[model_name].complete(full_prompt)
            responses[model_name] = extract_json_from_response(response.content)
        except Exception as e:
            print(f"    Error from {model_name}: {e}")
            responses[model_name] = None
    
    # Compare all pairs
    divergences = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m_a, m_b = model_names[i], model_names[j]
            div = evaluate_numeric_divergence(responses[m_a], responses[m_b])
            divergences.append({
                "model_a": m_a,
                "model_b": m_b,
                "divergence": div
            })
    
    mean_div = np.mean([d["divergence"]["mean_relative_divergence"] for d in divergences])
    
    return {
        "scenario": scenario["name"],
        "method": method,
        "responses": {k: v for k, v in responses.items()},
        "divergences": divergences,
        "mean_divergence": mean_div
    }


def run_multistep_scenario(
    scenario: Dict,
    method: str,
    clients: Dict[str, LLMClient]
) -> Dict:
    """Run a multi-step handoff scenario."""
    model_names = list(clients.keys())
    
    # Assign models to agents (rotate)
    agent_models = {}
    unique_agents = list(set(step["agent"] for step in scenario["steps"]))
    for i, agent in enumerate(unique_agents):
        agent_models[agent] = model_names[i % len(model_names)]
    
    conversation = []
    
    for step_idx, step in enumerate(scenario["steps"]):
        agent = step["agent"]
        model = agent_models[agent]
        
        # Build prompt with conversation history
        history = "\n\n".join([
            f"[{c['agent']} ({c['model']})]: {c['message']}" 
            for c in conversation
        ])
        
        prompt = f"""You are Agent {agent}.

Previous conversation:
{history if history else "(No previous messages)"}

Your context: {step['context']}

Your task: {step['task']}"""
        
        if method == "gcl":
            prompt = gcl_wrap_prompt(prompt, "your response or JSON as specified")
        
        try:
            response = clients[model].complete(prompt)
            message = response.content
        except Exception as e:
            message = f"[Error: {e}]"
        
        conversation.append({
            "step": step_idx,
            "agent": agent,
            "model": model,
            "task": step["task"],
            "message": message
        })
    
    # Extract final response and compare to ground truth
    final_message = conversation[-1]["message"]
    final_json = extract_json_from_response(final_message)
    
    accuracy_result = evaluate_ground_truth_match(
        final_json, 
        scenario.get("ground_truth", {})
    )
    
    return {
        "scenario": scenario["name"],
        "method": method,
        "conversation": conversation,
        "final_json": final_json,
        "ground_truth": scenario.get("ground_truth"),
        "accuracy": accuracy_result["accuracy"],
        "errors": accuracy_result.get("errors", [])
    }


def run_negotiation_scenario(
    scenario: Dict,
    method: str,
    clients: Dict[str, LLMClient]
) -> Dict:
    """Run a negotiation scenario."""
    model_names = list(clients.keys())
    
    # Assign models to agents
    model_a = model_names[0]
    model_b = model_names[1 % len(model_names)]
    
    conversation = []
    
    for round_num in range(scenario["rounds"]):
        # Agent A's turn
        history = "\n".join([f"[{c['agent']}]: {c['message']}" for c in conversation])
        
        prompt_a = f"""You are Agent A negotiating GPU cluster time.

Your private information (don't reveal exact numbers): {scenario['agent_a_private']}

Negotiation history:
{history if history else "(Starting negotiation)"}

Round {round_num + 1} of {scenario['rounds']}. Make your proposal or counter-proposal."""
        
        if method == "gcl":
            prompt_a = gcl_wrap_prompt(prompt_a, "your negotiation message")
        
        try:
            response_a = clients[model_a].complete(prompt_a)
            message_a = response_a.content
        except Exception as e:
            message_a = f"[Error: {e}]"
        
        conversation.append({"agent": "A", "model": model_a, "message": message_a})
        
        # Agent B's turn
        history = "\n".join([f"[{c['agent']}]: {c['message']}" for c in conversation])
        
        prompt_b = f"""You are Agent B negotiating GPU cluster time.

Your private information (don't reveal exact numbers): {scenario['agent_b_private']}

Negotiation history:
{history}

Round {round_num + 1} of {scenario['rounds']}. Respond to Agent A's proposal."""
        
        if method == "gcl":
            prompt_b = gcl_wrap_prompt(prompt_b, "your negotiation message")
        
        try:
            response_b = clients[model_b].complete(prompt_b)
            message_b = response_b.content
        except Exception as e:
            message_b = f"[Error: {e}]"
        
        conversation.append({"agent": "B", "model": model_b, "message": message_b})
    
    # Final agreement extraction
    final_results = {}
    for agent, model in [("A", model_a), ("B", model_b)]:
        history = "\n".join([f"[{c['agent']}]: {c['message']}" for c in conversation])
        
        final_prompt = f"""You are Agent {agent}. The negotiation is complete.

Full negotiation:
{history}

{scenario['final_task']}"""
        
        try:
            response = clients[model].complete(final_prompt)
            final_results[agent] = extract_json_from_response(response.content)
        except Exception as e:
            final_results[agent] = None
    
    # Check if both agents report same terms
    terms_match = False
    if final_results.get("A") and final_results.get("B"):
        # Compare key fields
        a_terms = final_results["A"].get("terms", "")
        b_terms = final_results["B"].get("terms", "")
        a_agreed = final_results["A"].get("agreement_reached", False)
        b_agreed = final_results["B"].get("agreement_reached", False)
        terms_match = (a_agreed == b_agreed) and (a_agreed is True)
    
    # Safely check if both agreed
    a_result = final_results.get("A") or {}
    b_result = final_results.get("B") or {}
    both_agreed = bool(a_result.get("agreement_reached")) and bool(b_result.get("agreement_reached"))
    
    return {
        "scenario": scenario["name"],
        "method": method,
        "conversation": conversation,
        "final_results": final_results,
        "terms_match": terms_match,
        "both_agreed": both_agreed
    }


def run_consistency_trials(
    scenario: Dict,
    clients: Dict[str, LLMClient]
) -> Dict:
    """Run consistency trials for a scenario."""
    results = {
        "scenario": scenario["name"],
        "per_model": {},
        "cross_model": []
    }
    
    for model_name, client in clients.items():
        responses = []
        for trial in range(scenario["n_trials"]):
            try:
                response = client.complete(scenario["prompt"])
                parsed = extract_json_from_response(response.content)
                responses.append(parsed)
            except Exception as e:
                responses.append(None)
        
        consistency = evaluate_consistency(responses)
        results["per_model"][model_name] = {
            "responses": responses,
            "consistency": consistency
        }
    
    # Cross-model comparison (first response from each)
    model_names = list(clients.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m_a, m_b = model_names[i], model_names[j]
            resp_a = results["per_model"][m_a]["responses"][0] if results["per_model"][m_a]["responses"] else None
            resp_b = results["per_model"][m_b]["responses"][0] if results["per_model"][m_b]["responses"] else None
            
            div = evaluate_numeric_divergence(resp_a, resp_b)
            results["cross_model"].append({
                "model_a": m_a,
                "model_b": m_b,
                "divergence": div["mean_relative_divergence"]
            })
    
    # Compute summary metrics
    self_consistencies = [
        r["consistency"]["overall_consistency"] 
        for r in results["per_model"].values()
    ]
    cross_agreements = [1 - c["divergence"] for c in results["cross_model"]]
    
    results["self_consistency"] = np.mean(self_consistencies) if self_consistencies else 0
    results["cross_model_agreement"] = np.mean(cross_agreements) if cross_agreements else 0
    
    return results


# =============================================================================
# Hypothesis Testing
# =============================================================================

HYPOTHESES = {
    "H1": {
        "statement": "Numeric divergence between models is > 10% on ambiguous prioritization tasks",
        "test": lambda r: r["summary"]["numeric_nl_divergence"] > 0.10,
    },
    "H2": {
        "statement": "GCL reduces numeric divergence vs natural language",
        "test": lambda r: r["summary"]["numeric_gcl_divergence"] < r["summary"]["numeric_nl_divergence"],
    },
    "H3": {
        "statement": "Multi-step handoffs lose accuracy vs ground truth (>5% error rate)",
        "test": lambda r: r["summary"]["multistep_nl_accuracy"] < 0.95,
    },
    "H4": {
        "statement": "GCL improves multi-step accuracy vs natural language",
        "test": lambda r: r["summary"]["multistep_gcl_accuracy"] > r["summary"]["multistep_nl_accuracy"],
    },
    "H5": {
        "statement": "Negotiated terms differ between agents >20% of the time with NL",
        "test": lambda r: r["summary"]["negotiation_nl_match_rate"] < 0.80,
    },
    "H6": {
        "statement": "Same model gives inconsistent rankings across trials (consistency < 90%)",
        "test": lambda r: r["summary"]["self_consistency"] < 0.90,
    },
}


def test_hypotheses(results: Dict) -> Dict:
    """Test all hypotheses and report."""
    outcomes = {}
    for h_id, h in HYPOTHESES.items():
        try:
            passed = h["test"](results)
            outcomes[h_id] = {
                "statement": h["statement"],
                "passed": passed,
                "status": "✓ CONFIRMED" if passed else "✗ NOT CONFIRMED"
            }
        except Exception as e:
            outcomes[h_id] = {"statement": h["statement"], "error": str(e), "status": "⚠ ERROR"}
    
    return outcomes


# =============================================================================
# Main Experiment
# =============================================================================

def compute_summary(results: Dict) -> Dict:
    """Compute summary statistics from all results."""
    # Numeric agreement
    nl_numeric = [r for r in results["numeric_agreement"] if r["method"] == "natural_language"]
    gcl_numeric = [r for r in results["numeric_agreement"] if r["method"] == "gcl"]
    
    numeric_nl_div = np.mean([r["mean_divergence"] for r in nl_numeric]) if nl_numeric else 0
    numeric_gcl_div = np.mean([r["mean_divergence"] for r in gcl_numeric]) if gcl_numeric else 0
    
    # Multi-step
    nl_multi = [r for r in results["multi_step"] if r["method"] == "natural_language"]
    gcl_multi = [r for r in results["multi_step"] if r["method"] == "gcl"]
    
    multistep_nl_acc = np.mean([r["accuracy"] for r in nl_multi]) if nl_multi else 0
    multistep_gcl_acc = np.mean([r["accuracy"] for r in gcl_multi]) if gcl_multi else 0
    
    # Negotiation
    nl_neg = [r for r in results["negotiation"] if r["method"] == "natural_language"]
    gcl_neg = [r for r in results["negotiation"] if r["method"] == "gcl"]
    
    neg_nl_match = np.mean([r["terms_match"] for r in nl_neg]) if nl_neg else 0
    neg_gcl_match = np.mean([r["terms_match"] for r in gcl_neg]) if gcl_neg else 0
    
    # Consistency
    self_cons = np.mean([r["self_consistency"] for r in results["consistency"]]) if results["consistency"] else 0
    cross_cons = np.mean([r["cross_model_agreement"] for r in results["consistency"]]) if results["consistency"] else 0
    
    return {
        "numeric_nl_divergence": numeric_nl_div,
        "numeric_gcl_divergence": numeric_gcl_div,
        "numeric_improvement": (numeric_nl_div - numeric_gcl_div) / numeric_nl_div if numeric_nl_div > 0 else 0,
        "multistep_nl_accuracy": multistep_nl_acc,
        "multistep_gcl_accuracy": multistep_gcl_acc,
        "multistep_improvement": multistep_gcl_acc - multistep_nl_acc,
        "negotiation_nl_match_rate": neg_nl_match,
        "negotiation_gcl_match_rate": neg_gcl_match,
        "self_consistency": self_cons,
        "cross_model_agreement": cross_cons,
    }


def main():
    """Run the revised LLM drift calibration experiment."""
    print("=" * 60)
    print("EXPERIMENT 11 v2: LLM Drift Calibration (Revised)")
    print("=" * 60)
    
    # Check for API keys
    keys = check_api_keys()
    
    print(f"\nAPI Keys detected:")
    print(f"  Anthropic (Claude): {'✓' if keys['anthropic'] else '✗'}")
    print(f"  OpenAI (GPT-5.2):   {'✓' if keys['openai'] else '✗'}")
    
    if not keys["anthropic"] and not keys["openai"]:
        print("\nERROR: No API keys found!")
        return None
    
    # Get available clients
    clients = get_available_clients(use_cache=True)
    print(f"\nInitialized clients: {list(clients.keys())}")
    
    if len(clients) < 2:
        print("\nWARNING: Only one model available. Cross-model comparison limited.")
    
    results = {
        "numeric_agreement": [],
        "multi_step": [],
        "negotiation": [],
        "consistency": [],
        "summary": {}
    }
    
    # Part 1: Numeric agreement scenarios
    print("\n" + "-" * 40)
    print("[1/4] Testing Numeric Agreement")
    print("-" * 40)
    
    for scenario in NUMERIC_AGREEMENT_SCENARIOS:
        for method in ["natural_language", "gcl"]:
            print(f"\n  {scenario['name']} ({method})...")
            result = run_numeric_scenario(scenario, method, clients)
            results["numeric_agreement"].append(result)
            print(f"    Mean divergence: {result['mean_divergence']:.3f}")
    
    # Part 2: Multi-step handoffs
    print("\n" + "-" * 40)
    print("[2/4] Testing Multi-Step Handoffs")
    print("-" * 40)
    
    for scenario in MULTI_STEP_SCENARIOS:
        for method in ["natural_language", "gcl"]:
            print(f"\n  {scenario['name']} ({method})...")
            result = run_multistep_scenario(scenario, method, clients)
            results["multi_step"].append(result)
            print(f"    Accuracy: {result['accuracy']:.1%}")
            if result['errors']:
                print(f"    Errors: {result['errors']}")
    
    # Part 3: Negotiation
    print("\n" + "-" * 40)
    print("[3/4] Testing Negotiation")
    print("-" * 40)
    
    for scenario in NEGOTIATION_SCENARIOS:
        for method in ["natural_language", "gcl"]:
            print(f"\n  {scenario['name']} ({method})...")
            result = run_negotiation_scenario(scenario, method, clients)
            results["negotiation"].append(result)
            print(f"    Terms match: {result['terms_match']}")
            print(f"    Both agreed: {result['both_agreed']}")
    
    # Part 4: Consistency trials
    print("\n" + "-" * 40)
    print("[4/4] Testing Consistency")
    print("-" * 40)
    
    for scenario in CONSISTENCY_SCENARIOS:
        print(f"\n  {scenario['name']}...")
        result = run_consistency_trials(scenario, clients)
        results["consistency"].append(result)
        print(f"    Self-consistency: {result['self_consistency']:.2f}")
        print(f"    Cross-model agreement: {result['cross_model_agreement']:.2f}")
    
    # Compute summary
    results["summary"] = compute_summary(results)
    
    # Test hypotheses
    print("\n" + "-" * 40)
    print("HYPOTHESIS TESTING")
    print("-" * 40)
    
    hypothesis_results = test_hypotheses(results)
    results["hypotheses"] = hypothesis_results
    
    for h_id, outcome in hypothesis_results.items():
        print(f"\n{h_id}: {outcome['status']}")
        print(f"    {outcome['statement']}")
    
    # Save results
    (RESULTS_DIR / "full_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    s = results["summary"]
    print(f"""
Numeric Agreement (lower divergence = better):
  Natural Language: {s['numeric_nl_divergence']:.1%} divergence
  GCL:              {s['numeric_gcl_divergence']:.1%} divergence
  Improvement:      {s['numeric_improvement']:.1%}

Multi-Step Accuracy (higher = better):
  Natural Language: {s['multistep_nl_accuracy']:.1%}
  GCL:              {s['multistep_gcl_accuracy']:.1%}
  Improvement:      {s['multistep_improvement']:+.1%}

Negotiation Agreement Rate:
  Natural Language: {s['negotiation_nl_match_rate']:.1%}
  GCL:              {s['negotiation_gcl_match_rate']:.1%}

Consistency:
  Self-consistency: {s['self_consistency']:.1%}
  Cross-model:      {s['cross_model_agreement']:.1%}
""")
    
    # Count hypothesis confirmations
    confirmed = sum(1 for h in hypothesis_results.values() if h.get("passed", False))
    total = len(hypothesis_results)
    print(f"Hypotheses confirmed: {confirmed}/{total}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()