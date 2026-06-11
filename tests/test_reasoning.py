"""Tests for Tree-of-Thought reasoning helpers."""

from types import SimpleNamespace

from biobank_agent.reasoning import ThoughtPath, ThoughtTree, ThoughtTreeResult, is_branching_question


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, max_tokens=0):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response)


def test_thought_path_and_result_summaries():
    path = ThoughtPath(
        approach="Metabolic panel",
        steps=["prevalence", "train_model"],
        strengths=["interpretable"],
        risks=["confounding"],
        score=0.75,
    )
    result = ThoughtTreeResult(
        question="Best E11 approach?",
        paths=[path],
        recommended=path,
        consensus="Check prevalence first",
    )

    assert "Metabolic panel" in path.summary()
    assert "score: 0.75" in path.summary()
    assert "Recommended" in result.summary()
    assert "Check prevalence" in result.summary()

    plain = ThoughtTreeResult(question="One path", paths=[path])
    assert "Consensus" not in plain.summary()
    assert "Recommended" not in plain.summary()


def test_explore_generates_scores_sorts_and_finds_consensus():
    llm = FakeLLM([
        """```json
        [
          {"approach": "Biomarker model", "steps": ["Check prevalence", "Train model"], "strengths": ["fast"], "risks": ["bias"]},
          {"approach": "Literature-first", "steps": ["Check prevalence", "Read papers"], "strengths": ["grounded"], "risks": ["slow"]}
        ]
        ```""",
        """[
          {"path": 1, "score": 0.65, "rationale": "feasible"},
          {"path": 2, "score": 0.90, "rationale": "more rigorous"}
        ]""",
    ])
    tree = ThoughtTree(llm=llm, max_branches=3)

    result = tree.explore(
        "What is the best way to analyze E11?",
        context="UKB data available",
        available_skills=["prevalence", "train_model"],
    )

    assert [p.approach for p in result.paths] == ["Literature-first", "Biomarker model"]
    assert result.recommended.approach == "Literature-first"
    assert "Check prevalence" in result.consensus
    assert len(llm.calls) == 2


def test_explore_returns_empty_result_when_generation_fails():
    tree = ThoughtTree(FakeLLM(["not json"]))

    result = tree.explore("best model?")

    assert result.paths == []
    assert result.recommended is None

    non_list = ThoughtTree(FakeLLM(['{"approach": "not a list"}']))
    assert non_list.explore("best model?").paths == []

    fenced_non_json = ThoughtTree(FakeLLM(["```text\nnot json\n```"]))
    assert fenced_non_json.explore("best model?").paths == []


def test_evaluation_failure_assigns_equal_scores():
    llm = FakeLLM([
        """[
          {"approach": "A", "steps": ["Check prevalence"], "strengths": [], "risks": []},
          {"approach": "B", "steps": ["Read papers"], "strengths": [], "risks": []}
        ]""",
        ValueError("review unavailable"),
    ])

    result = ThoughtTree(llm).explore("Which approach is better?")

    assert [p.score for p in result.paths] == [0.5, 0.5]


def test_evaluation_fenced_non_json_assigns_equal_scores():
    llm = FakeLLM([
        """[
          {"approach": "A", "steps": ["Check prevalence"], "strengths": [], "risks": []},
          {"approach": "B", "steps": ["Read papers"], "strengths": [], "risks": []}
        ]""",
        "```text\nnot json\n```",
    ])

    result = ThoughtTree(llm).explore("Which approach is better?")

    assert [p.score for p in result.paths] == [0.5, 0.5]


def test_evaluation_accepts_code_block_and_ignores_out_of_range_scores():
    llm = FakeLLM([
        """[
          {"approach": "A", "steps": ["Check prevalence"], "strengths": [], "risks": []},
          {"approach": "B", "steps": ["Read papers"], "strengths": [], "risks": []}
        ]""",
        """```json
        [
          {"path": 3, "score": 0.99, "rationale": "ignored"},
          {"path": 1, "score": 0.75, "rationale": "covered"}
        ]
        ```""",
    ])

    result = ThoughtTree(llm).explore("Which approach is better?")

    assert result.paths[0].approach == "A"
    assert result.paths[0].score == 0.75
    assert result.paths[1].score == 0.0


def test_find_consensus_handles_single_or_empty_paths():
    tree = ThoughtTree(FakeLLM([]))

    assert tree._find_consensus([]) == ""
    assert tree._find_consensus([ThoughtPath("A", ["one"], [], [])]) == ""
    assert tree._find_consensus([
        ThoughtPath("A", [], [], []),
        ThoughtPath("B", [], [], []),
    ]) == ""


def test_branching_question_detection():
    assert is_branching_question("Which model should we use for E11 prediction?")
    assert is_branching_question("Compare XGBoost vs LightGBM for biomarkers")
    assert is_branching_question("比较 两种 方法")
    assert not is_branching_question("Calculate prevalence for E11")
