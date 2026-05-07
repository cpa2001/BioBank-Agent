"""HealthFormer-style trajectory tokenization skill."""

import json

import pandas as pd

from biobank_agent.data.trajectory import TrajectoryTokenizer
from biobank_agent.registry import skill


@skill(
    name="trajectory_tokenize",
    description=(
        "Tokenize longitudinal multimodal participant measurements into HealthFormer-style "
        "time-ordered sequences. This prepares cohort-aligned evaluation data; it does not train a model."
    ),
    parameters={
        "rows_json": {
            "type": "string",
            "description": (
                "JSON list of rows with participant_id, timestamp, modality, value, optional "
                "value_type/unit/sleep. If empty, uses ctx.state.custom_data['trajectory_rows']."
            ),
            "default": "",
        },
        "max_bins": {
            "type": "integer",
            "description": "Maximum quantile bins per continuous modality",
            "default": 20,
        },
        "target_modality": {
            "type": "string",
            "description": "Optional future query modality to include in the output",
            "default": "",
        },
        "target_timestamp": {
            "type": "string",
            "description": "Optional future query timestamp for target_modality",
            "default": "",
        },
    },
    required=[],
)
def trajectory_tokenize(
    rows_json: str = "",
    max_bins: int = 20,
    target_modality: str = "",
    target_timestamp: str = "",
    *,
    ctx=None,
) -> dict:
    rows = None
    if rows_json:
        rows = json.loads(rows_json)
    elif ctx is not None:
        rows = getattr(getattr(ctx, "state", None), "custom_data", {}).get("trajectory_rows")
    if not rows:
        return {
            "error": "No trajectory rows supplied. Provide rows_json or ctx.state.custom_data['trajectory_rows'].",
            "expected_columns": ["participant_id", "timestamp", "modality", "value"],
        }

    df = pd.DataFrame(rows)
    tokenizer = TrajectoryTokenizer(max_bins=max_bins)
    dataset = tokenizer.fit(df).transform(df)
    future_query = None
    if target_modality and target_timestamp:
        future_query = tokenizer.build_future_query(target_modality, target_timestamp)

    if ctx is not None and hasattr(getattr(ctx, "memory", None), "upsert_node"):
        node_id = f"trajectory:{dataset.n_participants}:{dataset.n_tokens}"
        ctx.memory.upsert_node(
            "trajectory_dataset",
            node_id,
            payload={
                "n_participants": dataset.n_participants,
                "n_tokens": dataset.n_tokens,
                "vocab_size": dataset.vocab_size,
                "modalities": list(dataset.vocab.keys()),
                "missingness_by_modality": dataset.missingness_by_modality,
            },
            score=1.0,
        )

    return {
        "n_participants": dataset.n_participants,
        "n_tokens": dataset.n_tokens,
        "vocab_size": dataset.vocab_size,
        "modalities": list(dataset.vocab.keys()),
        "missingness_by_modality": dataset.missingness_by_modality,
        "first_sequence": dataset.sequences[0].to_dict() if dataset.sequences else {},
        "future_query": future_query,
        "status": "READY" if dataset.n_tokens else "PARTIAL",
    }
