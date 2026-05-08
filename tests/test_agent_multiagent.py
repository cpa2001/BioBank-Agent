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

    def test_force_strategy_for_execution_and_plain_query(self):
        from biobank_agent.agent import Agent
        from biobank_agent.complexity import Strategy

        agent = Agent.__new__(Agent)
        assert agent._forced_strategy("[PLAN MODE - Status: EXECUTION]\nRun it") == Strategy.ENSEMBLE
        assert agent._forced_strategy("normal query") is None

    def test_split_model_csv_trims_empty_values(self):
        from biobank_agent.agent import Agent

        assert Agent._split_model_csv(" a, ,b,, c ") == ["a", "b", "c"]

    def test_build_model_pool_falls_back_to_available_models(self):
        from biobank_agent.agent import Agent
        from biobank_agent.config import Settings

        settings = Settings(
            llm_model="default-model",
            model_pool="",
            preferred_multi_models="missing-a,missing-b",
            max_auto_model_pool=2,
        )

        pool = Agent._build_model_pool(
            settings=settings,
            available_models=["default-model", "backup-a", "backup-b"],
        )

        assert [spec.model_id for spec in pool] == ["default-model", "backup-a"]

    def test_switch_model_updates_client_and_pool(self):
        from biobank_agent.agent import Agent

        agent = Agent.__new__(Agent)
        agent.settings = type("Settings", (), {"llm_model": "old"})()
        agent.llm = type("LLM", (), {"model": "old"})()
        agent.orchestrator = type("Orch", (), {"default_model": "old", "model_pool": []})()

        agent.switch_model("new-model")

        assert agent.settings.llm_model == "new-model"
        assert agent.llm.model == "new-model"
        assert agent.orchestrator.default_model == "new-model"
        assert agent.orchestrator.model_pool[0].model_id == "new-model"
