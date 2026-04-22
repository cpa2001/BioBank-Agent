"""Tests for dynamic multi-agent model pool behavior."""


class TestModelPoolBuild:
    """Agent model pool should support discovery-driven defaults."""

    def test_auto_pool_uses_preferred_models_when_available(self):
        from biobank_agent.agent import Agent
        from biobank_agent.config import Settings

        settings = Settings(
            llm_model="claude-sonnet-4-6",
            model_pool="",
            preferred_multi_models="gpt-5.4,gemini-3.1-pro-preview",
            max_auto_model_pool=3,
        )

        pool = Agent._build_model_pool(
            settings=settings,
            available_models=["gpt-5.4", "gemini-3.1-pro-preview", "other-model"],
        )
        ids = [spec.model_id for spec in pool]

        assert ids[0] == "claude-sonnet-4-6"
        assert "gpt-5.4" in ids
        assert "gemini-3.1-pro-preview" in ids
        assert len(ids) == 3

    def test_explicit_model_pool_takes_priority(self):
        from biobank_agent.agent import Agent
        from biobank_agent.config import Settings

        settings = Settings(
            llm_model="claude-sonnet-4-6",
            model_pool="gpt-5.4,gemini-3.1-pro-preview",
            max_auto_model_pool=4,
        )

        pool = Agent._build_model_pool(
            settings=settings,
            available_models=["only-a", "only-b"],
        )
        ids = [spec.model_id for spec in pool]

        assert ids == [
            "claude-sonnet-4-6",
            "gpt-5.4",
            "gemini-3.1-pro-preview",
        ]


class TestPlanRoutingHints:
    """Plan mode context should force collaborative routing."""

    def test_extract_plan_mode_status(self):
        from biobank_agent.agent import Agent

        status = Agent._plan_mode_status("[PLAN MODE - Status: INTAKE]\nCurrent plan...")
        assert status == "INTAKE"

    def test_force_strategy_for_intake(self):
        from biobank_agent.agent import Agent
        from biobank_agent.complexity import Strategy

        agent = Agent.__new__(Agent)
        strategy = agent._forced_strategy("[PLAN MODE - Status: ALIGNMENT]\nUser says: refine scope")
        assert strategy == Strategy.SUPERVISOR
