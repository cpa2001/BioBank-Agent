# CLI Command Plugins

Biobank Agent slash commands are loaded from importable Python modules. Built-in
commands live under `biobank_agent.cli.commands`; local or third-party command
plugins can be added without editing the legacy REPL.

## Command Module Contract

A command module must expose a `commands()` factory returning
`RegisteredCommand` objects:

```python
from biobank_agent.cli.commands.base import CommandContext, RegisteredCommand


def commands() -> list[RegisteredCommand]:
    def handle(ctx: CommandContext, arg: str) -> None:
        ctx.console.print(f"plugin received: {arg}")

    return [
        RegisteredCommand(
            name="/plugin-demo",
            usage="/plugin-demo <text>",
            description="Run a plugin command demo",
            handle=handle,
        )
    ]
```

Command modules should not import `biobank_agent.cli_legacy`. Use
`CommandContext.console` for display and `CommandContext.action(...)` only when
you intentionally call an action exposed by the host CLI.

## Loading Plugins

Set `BIOBANK_CLI_COMMAND_MODULES` to a comma-separated list of importable module
names:

```bash
export BIOBANK_CLI_COMMAND_MODULES=my_package.biobank_commands,local_commands
biobank
```

For installed packages, expose the module through normal Python packaging. For a
local file, run `biobank` with `PYTHONPATH` containing the directory:

```bash
PYTHONPATH=/path/to/plugin_dir \
BIOBANK_CLI_COMMAND_MODULES=local_commands \
biobank
```

## Safety Rules

- Command names must start with `/`.
- Duplicate command names fail fast during registry construction.
- Plugin commands run in the same Python process as the CLI, so they should be
  treated as trusted local extensions.
- Long-running commands should emit concise status messages or delegate work to
  existing skills so the operator can see progress.

## Verification

Before relying on a plugin in a long run, verify discovery from the repository
root:

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from biobank_agent.cli.commands.registry import build_core_registry

registry = build_core_registry()
print(sorted(name for name in registry if name.startswith("/plugin")))
PY
```

Then run the CLI and type the new slash command directly.
