"""Workflow pattern analysis for optimizing skill sequencing.

Analyzes historical execution traces to:
- Identify successful skill sequences
- Detect resource bottlenecks
- Recommend parameter improvements
- Predict skill compatibility
"""

import logging
from typing import Optional, List, Dict, Any
from collections import defaultdict, Counter

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


def _analyze_skill_sequences(records: list) -> Dict[str, Any]:
    """Analyze patterns in skill execution sequences.
    
    Returns:
    - Frequent skill sequences (2-3 grams)
    - Success/failure rates per skill
    - Parameter correlations with success
    """
    if not records or len(records) < 2:
        return {
            "sequences": [],
            "skill_success_rates": {},
            "recommendations": ["Record more successful pipelines for better analysis"],
        }
    
    # Build skill sequences
    skill_names = [r.skill for r in records]
    bigrams = list(zip(skill_names, skill_names[1:]))
    trigrams = list(zip(skill_names, skill_names[1:], skill_names[2:])) if len(skill_names) > 2 else []
    
    # Count frequencies
    bigram_freq = Counter(bigrams).most_common(10)
    trigram_freq = Counter(trigrams).most_common(10)
    
    # Compute success metrics
    skill_success = defaultdict(lambda: {"success": 0, "total": 0})
    for record in records:
        skill_success[record.skill]["total"] += 1
        # If no error in interpretation, assume success
        if "error" not in record.interpretation.lower():
            skill_success[record.skill]["success"] += 1
    
    success_rates = {}
    for skill, counts in skill_success.items():
        # Items only exist after a record has incremented total.
        if counts["total"] > 0:  # pragma: no branch
            success_rates[skill] = round(counts["success"] / counts["total"] * 100, 1)
    
    # Generate recommendations
    recommendations = []
    # The early return above guarantees at least two records and therefore a bigram.
    if bigram_freq:  # pragma: no branch
        top_pattern = bigram_freq[0]
        recommendations.append(
            f"Top skill sequence: {top_pattern[0][0]} → {top_pattern[0][1]} "
            f"(observed {top_pattern[1]} times)"
        )
    
    # Identify low-success skills
    low_success = [
        (skill, rate) for skill, rate in success_rates.items() if rate < 80 and rate >= 0
    ]
    if low_success:
        worst = sorted(low_success, key=lambda x: x[1])[0]
        recommendations.append(
            f"Low success rate: '{worst[0]}' ({worst[1]}%). "
            f"Consider retry decorator or parameter adjustment."
        )
    
    return {
        "skill_sequences": {
            "bigrams": [{"sequence": b[0], "frequency": b[1]} for b in bigram_freq],
            "trigrams": [{"sequence": t[0], "frequency": t[1]} for t in trigram_freq],
        },
        "skill_success_rates": success_rates,
        "recommendations": recommendations,
    }


def _detect_resource_bottlenecks(records: list) -> Dict[str, Any]:
    """Detect resource-intensive operations and bottlenecks.
    
    Analyzes:
    - Skills that commonly fail with memory errors
    - Parameter values that correlate with failures
    - Data size transitions that trigger issues
    """
    if not records:
        return {"bottlenecks": [], "patterns": []}
    
    bottlenecks = []
    memory_issues = defaultdict(int)
    timeout_issues = defaultdict(int)
    
    # Track parameter values across executions
    parameter_patterns = defaultdict(list)
    
    for record in records:
        # Check for memory/resource errors in interpretation
        if "memory" in record.interpretation.lower():
            memory_issues[record.skill] += 1
        if "timeout" in record.interpretation.lower():
            timeout_issues[record.skill] += 1
        
        # Track parameter values
        for param_name, param_value in record.args.items():
            if isinstance(param_value, (int, float)):
                parameter_patterns[f"{record.skill}.{param_name}"].append({
                    "value": param_value,
                    "success": "error" not in record.interpretation.lower(),
                })
    
    # Identify bottlenecks
    for skill, count in memory_issues.items():
        bottlenecks.append({
            "type": "memory",
            "skill": skill,
            "occurrences": count,
            "recommendation": "Use @retry_on_error decorator to reduce complexity on retry",
        })
    
    for skill, count in timeout_issues.items():
        bottlenecks.append({
            "type": "timeout",
            "skill": skill,
            "occurrences": count,
            "recommendation": "Parallelize computation or reduce dataset size",
        })
    
    # Analyze parameter correlations
    patterns = []
    for param_key, values in parameter_patterns.items():
        if len(values) > 1:
            successes = sum(1 for v in values if v["success"])
            if successes < len(values):  # Has failures
                avg_success_val = sum(v["value"] for v in values if v["success"]) / max(1, successes)
                avg_fail_val = sum(v["value"] for v in values if not v["success"]) / max(1, len(values) - successes)
                if avg_fail_val > avg_success_val:
                    patterns.append({
                        "parameter": param_key,
                        "success_avg": round(avg_success_val, 2),
                        "failure_avg": round(avg_fail_val, 2),
                        "recommendation": f"Consider reducing {param_key.split('.')[-1]} below {int(avg_fail_val)}",
                    })
    
    return {
        "bottlenecks": bottlenecks,
        "parameter_patterns": patterns,
    }


def _predict_skill_compatibility(records: list) -> Dict[str, Any]:
    """Predict which skills can be chained together effectively.
    
    Analyzes output types and data flows to suggest compatible sequences.
    """
    if not records or len(records) < 3:
        return {
            "compatible_chains": [],
            "message": "Need more execution history for compatibility analysis",
        }
    
    # Build output→input flow graph
    skill_outputs = defaultdict(set)
    skill_inputs = defaultdict(set)
    
    for record in records:
        # Extract output types from key_results
        for key, value in record.key_results.items():
            if isinstance(value, (int, float)):
                skill_outputs[record.skill].add("numeric")
            elif isinstance(value, str):
                skill_outputs[record.skill].add("string")
            elif isinstance(value, dict):
                skill_outputs[record.skill].add("dict")
            elif isinstance(value, list):
                skill_outputs[record.skill].add("list")
        
        # Infer input types from args
        for key, value in record.args.items():
            if isinstance(value, (int, float)):
                skill_inputs[record.skill].add("numeric")
            elif isinstance(value, str):
                skill_inputs[record.skill].add("string")
            elif isinstance(value, dict):
                skill_inputs[record.skill].add("dict")
    
    # Find compatible chains
    compatible_chains = []
    skill_list = list(skill_outputs.keys())
    for i, skill1 in enumerate(skill_list):
        for skill2 in skill_list[i+1:]:
            # Check if skill1's output could feed into skill2's input
            common_types = skill_outputs[skill1] & skill_inputs[skill2]
            if common_types:
                compatible_chains.append({
                    "from": skill1,
                    "to": skill2,
                    "common_types": list(common_types),
                    "confidence": round(len(common_types) / max(len(skill_inputs[skill2]), 1), 2),
                })
    
    # Sort by confidence
    compatible_chains = sorted(compatible_chains, key=lambda x: x["confidence"], reverse=True)
    
    return {
        "compatible_chains": compatible_chains[:20],
        "total_chains": len(compatible_chains),
    }


@skill(
    name="analyze_workflow_patterns",
    description="Analyze execution patterns from session history to optimize skill sequencing. "
                "Identifies successful skill sequences, resource bottlenecks, parameter correlations, "
                "and compatible skill chains. Returns recommendations for pipeline optimization.",
    parameters={
        "analysis_depth": {
            "type": "string",
            "description": "Depth of analysis: 'quick' (sequences only) or 'detailed' (full analysis)",
            "default": "detailed",
        },
    },
    required=[],
)
def analyze_workflow_patterns(analysis_depth: str = "detailed", *, ctx=None) -> dict:
    """Analyze workflow patterns from execution history.
    
    This skill:
    1. Identifies frequently successful skill sequences
    2. Detects resource bottlenecks (memory, timeout issues)
    3. Correlates parameters with success/failure
    4. Predicts compatible skill chains for pipeline design
    """
    
    records = ctx.state.records
    
    if not records:
        return {
            "status": "no_data",
            "message": "No execution history yet. Run skills and try again.",
            "recommendations": [],
        }
    
    results = {
        "status": "success",
        "total_records": len(records),
        "records_analyzed": len(records),
    }
    
    # Sequence analysis (always included)
    sequence_analysis = _analyze_skill_sequences(records)
    results["skill_sequences"] = sequence_analysis["skill_sequences"]
    results["skill_success_rates"] = sequence_analysis["skill_success_rates"]
    
    if analysis_depth == "detailed":
        # Bottleneck analysis
        bottleneck_analysis = _detect_resource_bottlenecks(records)
        results["bottlenecks"] = bottleneck_analysis["bottlenecks"]
        results["parameter_patterns"] = bottleneck_analysis["parameter_patterns"]
        
        # Compatibility analysis
        compatibility = _predict_skill_compatibility(records)
        results["compatible_chains"] = compatibility["compatible_chains"]
    
    # Compile all recommendations
    all_recommendations = sequence_analysis["recommendations"]
    if analysis_depth == "detailed" and bottleneck_analysis.get("bottlenecks"):
        all_recommendations.extend([
            f"{b['skill']}: {b['recommendation']}" 
            for b in bottleneck_analysis["bottlenecks"][:3]
        ])
    
    results["recommendations"] = all_recommendations
    
    logger.info(
        f"Analyzed {len(records)} records. Found {len(results.get('bottlenecks', []))} bottlenecks, "
        f"{len(results.get('compatible_chains', []))} compatible chains"
    )
    
    return results


@skill(
    name="suggest_optimal_pipeline",
    description="Based on workflow pattern analysis, suggest an optimized pipeline for a given analysis goal. "
                "Returns recommended skill sequence with parameters.",
    parameters={
        "goal": {
            "type": "string",
            "description": "Analysis goal (e.g., 'disease_prediction', 'survival_analysis', 'prevalence_study')",
        },
        "max_steps": {
            "type": "integer",
            "description": "Maximum number of steps in suggested pipeline",
            "default": 5,
        },
    },
    required=["goal"],
)
def suggest_optimal_pipeline(goal: str, max_steps: int = 5, *, ctx=None) -> dict:
    """Suggest optimal skill pipeline based on historical patterns.
    
    Uses learned patterns to recommend a pipeline likely to succeed
    within resource constraints.
    """
    
    records = ctx.state.records
    memory = ctx.state.memory
    
    if not records:
        return {
            "status": "insufficient_data",
            "message": "Need execution history to suggest pipelines",
            "recommendation": "Run a few analysis pipelines first",
        }
    
    # Analyze patterns
    sequence_analysis = _analyze_skill_sequences(records)
    success_rates = sequence_analysis["skill_success_rates"]
    
    # Filter skills by success rate (> 70%)
    reliable_skills = [
        skill for skill, rate in success_rates.items() if rate >= 70
    ]
    
    if not reliable_skills:
        return {
            "status": "low_reliability",
            "message": "No highly reliable skills found in history",
            "recommendations": sequence_analysis["recommendations"],
        }
    
    # Map goals to typical skill sequences
    goal_patterns = {
        "disease_prediction": ["prevalence", "build_cohort", "train_model", "evaluate"],
        "survival_analysis": ["prevalence", "survival", "compare_groups"],
        "prevalence_study": ["prevalence", "list_errors"],
        "model_comparison": ["build_cohort", "train_model", "compare_models"],
    }
    
    suggested_skills = goal_patterns.get(goal, reliable_skills[:max_steps])
    
    # Filter to reliable skills and limit to max_steps
    pipeline = [s for s in suggested_skills if s in reliable_skills][:max_steps]
    
    # Get configs from long-term memory if available
    enriched_pipeline = []
    for skill_name in pipeline:
        step = {
            "skill": skill_name,
            "success_rate": success_rates.get(skill_name, 0),
        }
        
        # Check if there's a saved config for this skill
        for key, config in memory._data.get("model_configs", {}).items():
            if skill_name in key:
                step["suggested_config"] = config["config"]
                break
        
        enriched_pipeline.append(step)
    
    return {
        "status": "success",
        "goal": goal,
        "pipeline": enriched_pipeline,
        "total_steps": len(enriched_pipeline),
        "estimated_success_rate": round(
            sum(s["success_rate"] for s in enriched_pipeline) / len(enriched_pipeline) 
            if enriched_pipeline else 0, 1
        ),
        "recommendations": sequence_analysis["recommendations"],
    }
