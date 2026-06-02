"""Edge coverage for skill registry loading and discovery."""

import importlib
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from biobank_agent.registry import (
    SkillRegistry,
    autodiscover_skills,
    discover_custom_skills,
    get_registry,
    skill,
)


def _schema(name="unit_skill", description="Unit skill"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


def test_register_execute_injects_context_and_unregisters():
    reg = SkillRegistry()
    ctx = SimpleNamespace(marker="ctx")

    def echo(value, ctx=None):
        return {"value": value, "ctx": ctx.marker}

    reg.register("echo", echo, _schema("echo", "Echo values"))

    args = {"value": 3}
    result = reg.execute("echo", args, ctx=ctx)

    assert result == {"value": 3, "ctx": "ctx"}
    assert "ctx" not in args
    assert "echo" in reg
    assert len(reg) == 1
    assert reg.list_skills() == [{"name": "echo", "description": "Echo values"}]
    assert reg.tool_schemas()[0]["function"]["name"] == "echo"

    reg.unregister("echo")
    assert "echo" not in reg
    assert len(reg) == 0


def test_execute_lazy_skill_finds_decorated_function(monkeypatch):
    reg = SkillRegistry()
    schema = _schema("lazy_alias")

    def actual(value):
        return {"value": value}

    actual._skill_name = "lazy_alias"
    actual._skill_schema = schema
    module = SimpleNamespace(actual=actual)

    monkeypatch.setattr(importlib, "import_module", lambda _path: module)
    reg.register_lazy("lazy_alias", "fake.module", schema)

    assert reg.execute("lazy_alias", {"value": 7}) == {"value": 7}
    assert "lazy_alias" in reg._callables


def test_execute_lazy_skill_uses_direct_module_attribute(monkeypatch):
    reg = SkillRegistry()

    def lazy_direct(value):
        return {"value": value}

    module = SimpleNamespace(lazy_direct=lazy_direct)
    monkeypatch.setattr(importlib, "import_module", lambda _path: module)
    reg.register_lazy("lazy_direct", "fake.module", _schema("lazy_direct"))

    assert reg.execute("lazy_direct", {"value": 8}) == {"value": 8}


def test_execute_lazy_skill_raises_when_function_missing(monkeypatch):
    reg = SkillRegistry()
    reg.register_lazy("missing_skill", "fake.module", _schema("missing_skill"))
    monkeypatch.setattr(importlib, "import_module", lambda _path: SimpleNamespace())

    with pytest.raises(ValueError, match="not found"):
        reg.execute("missing_skill", {})


def test_execute_unknown_skill_raises():
    with pytest.raises(ValueError, match="Unknown skill"):
        SkillRegistry().execute("unknown", {})


def test_reload_module_skill_reregisters_decorated_callable(monkeypatch):
    reg = SkillRegistry()
    schema = _schema("reloadable")
    module = ModuleType("fake_reload_module")

    def replacement():
        return {"status": "reloaded"}

    replacement._skill_name = "reloadable"
    replacement._skill_schema = schema
    module.replacement = replacement

    monkeypatch.setattr(importlib, "import_module", lambda _path: module)
    monkeypatch.setattr(importlib, "reload", lambda mod: mod)

    reg.register_lazy("reloadable", "fake.reload_module", schema)
    reg.reload_skill("reloadable")

    assert reg.execute("reloadable", {}) == {"status": "reloaded"}


def test_reload_module_skill_tolerates_missing_schema_or_no_match(monkeypatch):
    reg = SkillRegistry()
    module = ModuleType("fake_reload_module")

    def no_schema():
        return {"status": "ignored"}

    no_schema._skill_name = "reloadable"
    module.no_schema = no_schema
    monkeypatch.setattr(importlib, "import_module", lambda _path: module)
    monkeypatch.setattr(importlib, "reload", lambda mod: mod)

    reg.register_lazy("reloadable", "fake.reload_module", _schema("reloadable"))
    reg.reload_skill("reloadable")
    assert "reloadable" not in reg._callables

    delattr(module.no_schema, "_skill_name")
    reg.reload_skill("reloadable")
    assert "reloadable" not in reg._callables


def test_reload_direct_callable_warns_and_unknown_raises(caplog):
    reg = SkillRegistry()
    reg.register("direct", lambda: {"ok": True}, _schema("direct"))

    with caplog.at_level(logging.WARNING):
        reg.reload_skill("direct")

    assert "cannot reload" in caplog.text
    with pytest.raises(ValueError, match="Unknown skill"):
        reg.reload_skill("absent")


def test_skill_decorator_infers_required_fields_and_registers_globally():
    name = "unit_required_defaults"
    registry = get_registry()
    registry.unregister(name)

    try:
        @skill(
            name=name,
            description="Decorator test",
            parameters={
                "required_arg": {"type": "string", "description": "Required"},
                "optional_arg": {"type": "integer", "description": "Optional", "default": 1},
            },
        )
        def decorated(required_arg, optional_arg=1):
            return {"required_arg": required_arg, "optional_arg": optional_arg}

        assert decorated._skill_name == name
        assert decorated._skill_schema["function"]["parameters"]["required"] == ["required_arg"]
        assert registry.execute(name, {"required_arg": "x"}) == {
            "required_arg": "x",
            "optional_arg": 1,
        }
    finally:
        registry.unregister(name)


def test_skill_decorator_strips_property_required_metadata():
    name = "unit_required_property_metadata"
    registry = get_registry()
    registry.unregister(name)

    try:
        @skill(
            name=name,
            description="Decorator metadata test",
            parameters={
                "required_arg": {"type": "string", "description": "Required", "required": True},
                "optional_arg": {"type": "integer", "description": "Optional", "required": False},
            },
        )
        def decorated(required_arg, optional_arg=None):
            return {"required_arg": required_arg, "optional_arg": optional_arg}

        params = decorated._skill_schema["function"]["parameters"]
        assert params["required"] == ["required_arg"]
        assert "required" not in params["properties"]["required_arg"]
        assert "required" not in params["properties"]["optional_arg"]
    finally:
        registry.unregister(name)


def test_autodiscover_missing_package_logs_warning(caplog):
    with caplog.at_level(logging.WARNING):
        autodiscover_skills("not_a_real_skill_package_for_tests")

    assert "Could not import not_a_real_skill_package_for_tests" in caplog.text


def test_autodiscover_skips_private_modules_and_logs_import_failures(tmp_path, monkeypatch, caplog):
    package = tmp_path / "fake_skill_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "_private.py").write_text("raise RuntimeError('should be skipped')\n", encoding="utf-8")
    (package / "good.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "bad.py").write_text("raise RuntimeError('broken import')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    with caplog.at_level(logging.WARNING):
        autodiscover_skills("fake_skill_pkg")

    assert "Failed to load skill fake_skill_pkg.bad" in caplog.text
    assert "fake_skill_pkg._private" not in sys.modules


def test_discover_custom_skills_handles_missing_private_bad_and_valid_files(tmp_path, caplog):
    registry = get_registry()
    registry.unregister("unit_custom_skill")
    (tmp_path / "_private.py").write_text("raise RuntimeError('skip me')\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("raise RuntimeError('broken custom')\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text(
        "from biobank_agent.registry import skill\n"
        "@skill(name='unit_custom_skill', description='Custom test', parameters={})\n"
        "def unit_custom_skill():\n"
        "    return {'loaded': True}\n",
        encoding="utf-8",
    )

    try:
        assert discover_custom_skills(tmp_path / "missing") == 0
        with caplog.at_level(logging.WARNING):
            loaded = discover_custom_skills(tmp_path)

        assert loaded == 1
        assert "Failed to load custom skill bad.py" in caplog.text
        assert registry.execute("unit_custom_skill", {}) == {"loaded": True}
    finally:
        registry.unregister("unit_custom_skill")
        sys.modules.pop("custom_skills.ok", None)
        sys.modules.pop("custom_skills.bad", None)


def test_discover_custom_skills_skips_files_without_import_specs(tmp_path, monkeypatch):
    (tmp_path / "nospec.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr("biobank_agent.registry.importlib.util.spec_from_file_location", lambda *args, **kwargs: None)

    assert discover_custom_skills(tmp_path) == 0
