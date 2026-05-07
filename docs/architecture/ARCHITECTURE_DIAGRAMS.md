# Biobank Agent: Skills Architecture - Visual Diagrams

## Diagram 1: Skill Discovery Flow at Agent Initialization

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent.__init__() called                          │
│              [biobank_agent/agent.py:63-120]                        │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ├─────────────────────────────────────────────────────┐
             │                                                     │
    ┌────────▼──────────────────┐                   ┌─────────────▼─────┐
    │ autodiscover_skills()      │                   │ discover_custom   │
    │ [registry.py:184-201]      │                   │ _skills()         │
    │                            │                   │ [registry.py:     │
    │ • Import "biobank_agent    │                   │  204-233]         │
    │   .skills" package         │                   │                   │
    │ • Scan all .py modules     │                   │ • Check if        │
    │ • Skip those starting      │                   │   ./custom_skills │
    │   with "_"                 │                   │   directory       │
    │                            │                   │   exists          │
    └────────┬──────────────────┘                   │                   │
             │                                       │ • For each .py    │
             │ ├─ importlib.import_module(          │   file:           │
             │ │  "biobank_agent.skills.prevalence" │   - Skip if       │
             │ │  "biobank_agent.skills.brainstorm" │     starts with   │
             │ │  ... (44 modules total)            │     "_"           │
             │ │                                     │   - Use           │
             │ └─ Each module runs on import        │     importlib     │
             │   (triggers @skill decorators)       │     .util.spec    │
             │                                       │   - Load module   │
    ┌────────▼────────────────────────────┐         │                   │
    │ @skill decorator executed            │         │ • sys.modules    │
    │ for each decorated function          │         │   ["custom_      │
    │                                       │         │    skills.*."]   │
    │ func._skill_name = "prevalence"      │         │                   │
    │ func._skill_schema = {...}            │         │ • exec_module()  │
    │ _registry.register(...)              │         │   triggers       │
    │                                       │         │   @skill         │
    └────────┬────────────────────────────┘         │   decorators     │
             │                                       │                   │
             │ (44 skills registered)               └──────────┬────────┘
             │                                                 │
             └──────────────────────┬──────────────────────────┘
                                    │
                          ┌─────────▼────────────┐
                          │ get_registry()       │
                          │ [registry.py:131]    │
                          │                      │
                          │ Returns global       │
                          │ _registry singleton  │
                          └─────────┬────────────┘
                                    │
                          ┌─────────▼────────────────────────┐
                          │ SkillRegistry populated with:     │
                          │                                   │
                          │ _schemas: name → schema           │
                          │ _callables: name → function       │
                          │ _module_paths: name → module path │
                          │ _descriptions: name → description │
                          │                                   │
                          │ Total: 44+ built-in + N custom    │
                          └────────────────────────────────────┘
```

---

## Diagram 2: Skill Invocation During Agent.run()

```
┌──────────────────────────────────────────────────────────────────┐
│ User Query: "analyze disease E11 for markers"                    │
└────────────┬─────────────────────────────────────────────────────┘
             │
    ┌────────▼────────────────────────────────────┐
    │ agent.run(user_query)                       │
    │ [agent.py:280-490]                          │
    │                                              │
    │ Enters tool_call loop                        │
    │ (max_tool_rounds=30)                         │
    └────────┬────────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────────────────────┐
    │ Build system message + call LLM                           │
    │ [agent.py:296-312]                                        │
    │                                                            │
    │ messages = [system_prompt + user_query + history]        │
    │ tools = self.registry.tool_schemas()  ◄─────────────┐     │
    │          [schema for 44+ skills]                    │     │
    │                                                      │     │
    │ response = llm.chat(messages, tools=tools)         │     │
    │                                                     │     │
    └────────┬──────────────────────────────────────────┤─────┘
             │                                          │
    ┌────────▼─────────────────────────┐               │
    │ LLM Response Generated            │               │
    │ (function calls in OpenAI format) │               │
    │                                   │               │
    │ [                                 │               │
    │   {                               │               │
    │     "id": "call_abc123",          │               │
    │     "type": "function",           │               │
    │     "function": {                 │               │
    │       "name": "prevalence",       │               │
    │       "arguments": {              │               │
    │         "top_n": 20,              │               │
    │         "chapter_filter": "E"     │               │
    │       }                           │               │
    │     }                             │               │
    │   }                               │               │
    │ ]                                 │               │
    └────────┬─────────────────────────┘               │
             │                                          │
    ┌────────▼──────────────────────────────────────────┼───────┐
    │ For each tool_call:                               │       │
    │                                                    │       │
    │ 1. Build context:                                 │       │
    │    ctx = self._build_ctx(report_dir)             │       │
    │    [agent.py:600-615]                            │       │
    │                                                    │       │
    │    ctx.dm = DataManager                          │       │
    │    ctx.catalog = FieldCatalog                    │       │
    │    ctx.state = SessionState                      │       │
    │    ctx.settings = Settings                       │       │
    │    ctx.memory = LongTermMemory                   │       │
    │    ctx.report_dir = Path                         │       │
    │                                                    │       │
    │ 2. Execute skill:                                │       │
    │    result = self.registry.execute(               │       │
    │        tc.name="prevalence",                     │       │
    │        tc.args={"top_n": 20, ...},              │       │
    │        ctx=ctx                                   │       │
    │    )                                             │       │
    │    [registry.py:82-104]                          │       │
    │                                                   │       │
    │    registry.execute() logic:                     │       │
    │    ├─ Look up function in _callables             │       │
    │    ├─ If lazy-loaded, import module              │       │
    │    ├─ Inject ctx as keyword argument             │       │
    │    └─ Call func(**args, ctx=ctx)                │       │
    │                                                   │       │
    └────────┬──────────────────────────────────────────┼───────┘
             │                                          │
    ┌────────▼─────────────────────────────────────────┤────┐
    │ Skill Function Executes                          │    │
    │ def prevalence(top_n, chapter_filter, *, ctx):  │    │
    │                                                   │    │
    │   dm = ctx.dm  ◄───────────────────────────────┤    │
    │   result = dm.query(...)                       │    │
    │   # ... process data ...                        │    │
    │   return {                                       │    │
    │     "total_subjects": 50000,                    │    │
    │     "top_diseases": [...],                      │    │
    │     "figure": "path/to/chart.png"               │    │
    │   }                                              │    │
    └────────┬────────────────────────────────────────┼────┘
             │ (returns dict)                          │
             │                                         │
    ┌────────▼─────────────────────────────────────────┼────┐
    │ Tool Result Processing                           │    │
    │ [agent.py:340-482]                               │    │
    │                                                   │    │
    │ • Convert result to JSON                          │    │
    │ • Catch and record errors in memory             │    │
    │ • Track execution time                           │    │
    │ • Record provenance                              │    │
    │ • Create AnalysisRecord in state                │    │
    │ • Run Reflexion engine if error                 │    │
    │ • Append result to message history              │    │
    │                                                   │    │
    │ messages.append({                                │    │
    │   "role": "tool",                               │    │
    │   "tool_call_id": "call_abc123",                │    │
    │   "content": json.dumps(result)                 │    │
    │ })                                               │    │
    │                                                   │    │
    └────────┬────────────────────────────────────────┼────┘
             │                                         │
    ┌────────▼──────────────────────────────────────┬─┘────┐
    │ Next Iteration or Final Response               │       │
    │                                                 │       │
    │ if response.has_tool_calls:                    │       │
    │   continue loop  ──────────────────────────────┘       │
    │ else:                                                   │
    │   append final text response                           │
    │   return response.text                                 │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
```

---

## Diagram 3: Registry Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              SkillRegistry (registry.py:29-124)                  │
│                     GLOBAL SINGLETON                            │
│                                                                  │
│  Created at module load:                                        │
│  _registry = SkillRegistry()                                    │
│  get_registry() → returns _registry                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Internal State:                                                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ _schemas: dict[name, schema]                              │ │
│  │ ─────────────────────────────────────────                 │ │
│  │ "prevalence" → {                                           │ │
│  │   "type": "function",                                      │ │
│  │   "function": {                                            │ │
│  │     "name": "prevalence",                                  │ │
│  │     "description": "Calculate disease prevalence...",     │ │
│  │     "parameters": {                                        │ │
│  │       "type": "object",                                    │ │
│  │       "properties": {                                      │ │
│  │         "top_n": {"type": "integer", "description": ...} │ │
│  │       },                                                   │ │
│  │       "required": []                                       │ │
│  │     }                                                      │ │
│  │   }                                                        │ │
│  │ }                                                          │ │
│  │                                                            │ │
│  │ → Used to generate tool_use prompts for LLM              │ │
│  │ → Given to: llm.chat(messages, tools=tool_schemas())     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ _callables: dict[name, Callable]                          │ │
│  │ ──────────────────────────────────────                    │ │
│  │ "prevalence" → <function prevalence at 0x...>            │ │
│  │ "think" → <function think at 0x...>                       │ │
│  │ ...                                                        │ │
│  │                                                            │ │
│  │ → Used to execute skills                                  │ │
│  │ → Looked up when: registry.execute("prevalence", ...)    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ _module_paths: dict[name, module_path]                    │ │
│  │ ───────────────────────────────────────                   │ │
│  │ "some_lazy_skill" → "biobank_agent.skills.xyz"           │ │
│  │                                                            │ │
│  │ → Used for lazy loading                                   │ │
│  │ → If skill not in _callables, import from path           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ _descriptions: dict[name, description]                    │ │
│  │ ────────────────────────────────────────                  │ │
│  │ "prevalence" → "Calculate disease prevalence in..."      │ │
│  │                                                            │ │
│  │ → Used for CLI /skills command                            │ │
│  │ → registry.list_skills() returns name + description      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Key Methods:                                                   │
│                                                                  │
│  register(name, func, schema)                                   │
│  ├─ Called by @skill decorator during module import           │ │
│  └─ Stores schema + callable (eager)                          │ │
│                                                                  │
│  register_lazy(name, module_path, schema)                      │ │
│  ├─ Register schema only                                       │ │
│  └─ Load implementation on first execute() call              │ │
│                                                                  │
│  execute(name, args, ctx=None)                                 │ │
│  ├─ Look up function in _callables                            │ │
│  ├─ Lazy-load if needed (via _module_paths)                  │ │
│  ├─ Inject ctx as keyword argument: args["ctx"] = ctx        │ │
│  └─ Call func(**args, ctx=ctx)                              │ │
│                                                                  │
│  tool_schemas() → list[dict]                                    │
│  └─ Returns _schemas.values()                                  │ │
│                                                                  │
│  list_skills() → list[{name, description}]                     │ │
│  └─ Used by CLI /skills command                               │ │
│                                                                  │
│  reload_skill(name)                                            │ │
│  └─ Hot-reload: re-import module + re-register               │ │
│                                                                  │
│  __len__() → int                                               │ │
│  └─ Number of registered skills                               │ │
│                                                                  │
│  __contains__(name) → bool                                     │ │
│  └─ Check if skill exists                                     │ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Diagram 4: @skill Decorator Flow

```
┌──────────────────────────────────────────────────────────────┐
│                   Skill File Loaded                          │
│              biobank_agent/skills/prevalence.py              │
│                                                              │
│  from biobank_agent.registry import skill                   │
│                                                              │
│  @skill(                                                     │
│      name="prevalence",                                      │
│      description="Calculate disease prevalence...",          │
│      parameters={                                            │
│          "top_n": {...},                                     │
│          "chapter_filter": {...},                            │
│      },                                                      │
│      required=[],                                            │
│  )                                                           │
│  def prevalence(top_n: int = 20, chapter_filter: str = "", │
│                 *, ctx=None) -> dict:                       │
│      ...                                                     │
│                                                              │
└────────────────────┬─────────────────────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────────────────┐
    │ @skill decorator called                              │
    │ [registry.py:137-179]                                │
    │                                                       │
    │ 1. Build OpenAI function-calling schema             │ │
    │    ├─ Extract properties from parameters dict       │ │
    │    ├─ Auto-detect required fields                   │ │
    │    │  (fields without "default" are required)       │ │
    │    └─ Create schema dict in OpenAI format           │ │
    │                                                       │
    │ 2. Create decorator function                         │ │
    │    ├─ Attach _skill_name attribute                  │ │
    │    ├─ Attach _skill_schema attribute                │ │
    │    └─ Call _registry.register(name, func, schema)  │ │
    │                                                       │
    │ 3. Return the function unchanged                    │ │
    │    (decorator is transparent)                        │ │
    │                                                       │
    └────────────────┬──────────────────────────────────────┘
                     │
    ┌────────────────▼──────────────────────────────────────┐
    │ _registry.register() called                          │
    │ [registry.py:40-44]                                  │
    │                                                       │
    │ _schemas["prevalence"] = schema_dict               │ │
    │ _callables["prevalence"] = func_object             │ │
    │ _descriptions["prevalence"] = description_string   │ │
    │                                                       │
    │ ✓ Skill is now registered                           │ │
    │ ✓ LLM can see it in tool_schemas()                 │ │
    │ ✓ Can be executed via registry.execute()           │ │
    │                                                       │
    └────────────────────────────────────────────────────────┘
```

---

## Diagram 5: Custom Skills Discovery Process

```
┌───────────────────────────────────────────────────────────────┐
│          Custom Skill Discovery at Agent Init                 │
│           discover_custom_skills(custom_dir)                  │
│              [registry.py:204-233]                            │
└───────────────┬───────────────────────────────────────────────┘
                │
    ┌───────────▼───────────────────────────────────────┐
    │ Check directory exists                            │
    │                                                   │
    │ custom_dir = Path("./custom_skills")             │
    │ (from config.py line 87)                          │
    │                                                   │
    │ if not custom_dir.exists():                       │
    │   return 0                                        │
    │                                                   │
    └───────────┬───────────────────────────────────────┘
                │
    ┌───────────▼───────────────────────────────────────┐
    │ Scan for .py files                                │
    │                                                   │
    │ for py_file in sorted(custom_dir.glob("*.py")): │
    │   (sorted for consistent loading order)           │
    │                                                   │
    │   if py_file.name.startswith("_"):               │
    │     skip  (private files)                         │
    │                                                   │
    │   module_name = f"custom_skills.{stem}"          │
    │   (e.g., "custom_skills.my_analysis")            │
    │                                                   │
    └───────────┬───────────────────────────────────────┘
                │
    ┌───────────▼───────────────────────────────────────────┐
    │ Dynamic Module Loading                                │
    │                                                       │
    │ spec = importlib.util.spec_from_file_location(      │ │
    │   "custom_skills.my_analysis",                      │ │
    │   "/path/to/custom_skills/my_analysis.py"          │ │
    │ )                                                    │ │
    │                                                       │
    │ mod = importlib.util.module_from_spec(spec)        │ │
    │ sys.modules["custom_skills.my_analysis"] = mod     │ │
    │                                                       │
    │ spec.loader.exec_module(mod)                        │ │
    │ ↓                                                    │ │
    │ [Module code runs, @skill decorators execute]      │ │
    │ ↓                                                    │ │
    │ skill_function._skill_name = "my_analysis"         │ │
    │ skill_function._skill_schema = {...}               │ │
    │ _registry.register(...)                            │ │
    │                                                       │
    │ ✓ Skill registered with global registry            │ │
    │                                                       │
    └───────────┬───────────────────────────────────────────┘
                │
    ┌───────────▼───────────────────────────────────────┐
    │ Continue for all .py files in directory           │
    │                                                   │
    │ Return count of newly loaded skills               │
    │                                                   │
    │ logger.info(                                      │ │
    │   "Loaded %d custom skill(s) from %s",           │ │
    │   loaded, custom_dir                             │ │
    │ )                                                 │ │
    │                                                   │
    └───────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────┐
│           File Structure Example                     │
│                                                      │
│  ./custom_skills/                                   │
│  ├── _internal.py          (SKIPPED)               │ │
│  ├── my_analysis.py        (LOADED)                │ │
│  ├── special_query.py      (LOADED)                │ │
│  └── helpers.py            (LOADED)                │ │
│                                                      │
│  Result: 3 custom skills + 44 built-in = 47 total  │ │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Diagram 6: Context Injection at Execution

```
┌────────────────────────────────────────────────────────────┐
│           registry.execute() Flow                         │
│     [registry.py:82-104]                                  │
└───────────────┬────────────────────────────────────────────┘
                │
    ┌───────────▼──────────────────────────────────┐
    │ execute(name="prevalence",                   │
    │         args={"top_n": 20, ...},            │
    │         ctx=SkillContext)                    │
    │                                              │
    │ 1. Look up function                          │
    │    func = _callables.get(name)              │
    │    if not found:                             │
    │    ├─ Check _module_paths for lazy load     │
    │    ├─ importlib.import_module(path)         │
    │    └─ Search for function with matching     │
    │       _skill_name attribute                  │
    │                                              │
    │ 2. Inject context                            │
    │    if ctx is not None:                      │
    │      args["ctx"] = ctx                       │
    │                                              │
    │ 3. Execute                                   │
    │    return func(**args)                       │
    │                                              │
    └───────────┬──────────────────────────────────┘
                │
    ┌───────────▼──────────────────────────────────────┐
    │ Function Call (with ctx injected)                │
    │                                                   │
    │ prevalence(top_n=20, ctx=SkillContext)          │
    │ │                                                 │
    │ └─> Inside function:                             │
    │     dm = ctx.dm              # DataManager        │
    │     settings = ctx.settings  # Settings           │
    │     state = ctx.state        # SessionState       │
    │     memory = ctx.memory      # LongTermMemory     │
    │     report_dir = ctx.report_dir  # Path           │
    │     catalog = ctx.catalog    # FieldCatalog       │
    │                                                   │
    │     # Implementation uses context                 │
    │     result = dm.query(...)                        │
    │     ...                                           │
    │     return {"status": "success", ...}             │
    │                                                   │
    └───────────┬──────────────────────────────────────┘
                │
    ┌───────────▼──────────────────────────────────────┐
    │ Return to Agent Loop                             │
    │                                                   │
    │ result_str = json.dumps(result)                  │
    │ messages.append({                                │
    │   "role": "tool",                               │
    │   "tool_call_id": "call_xyz",                   │
    │   "content": result_str                          │
    │ })                                               │
    │                                                   │
    │ continue loop  (LLM sees result)                 │
    │                                                   │
    └───────────────────────────────────────────────────┘
```

---

## Diagram 7: Message Flow Through Agent Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    Complete Agent Loop                          │
│                       agent.run()                               │
└──────────────────┬──────────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │ Round 1              │
        │ User: "analyze E11"  │
        │                      │
        │ messages = [         │
        │   {                  │
        │     role: "system",  │
        │     content: SYSTEM  │
        │   },                 │
        │   {                  │
        │     role: "user",    │
        │     content: query   │
        │   }                  │
        │ ]                    │
        │                      │
        │ tools = registry     │
        │   .tool_schemas()    │
        │   (44+ OpenAI        │
        │    schemas)          │
        │                      │
        │ response = llm.chat( │
        │   messages,          │
        │   tools              │
        │ )                    │
        │                      │
        │ response.tool_calls  │
        │   = [{               │
        │     name: "think",   │
        │     args: {...}      │
        │   }]                 │
        │                      │
        └──────────┬───────────┘
                   │ has_tool_calls=true
                   │
        ┌──────────▼──────────────────────┐
        │ Execute tool: "think"            │
        │                                  │
        │ result = registry.execute(       │
        │   "think",                       │
        │   {"reasoning": "..."},          │
        │   ctx=ctx                        │
        │ )                                │
        │                                  │
        │ messages.append({                │
        │   role: "assistant",             │
        │   content: null,                 │
        │   tool_calls: [{...}]            │
        │ })                               │
        │                                  │
        │ messages.append({                │
        │   role: "tool",                  │
        │   tool_call_id: "...",           │
        │   content: result_json           │
        │ })                               │
        │                                  │
        └──────────┬──────────────────────┘
                   │
        ┌──────────▼───────────┐
        │ Round 2              │
        │ (same flow)          │
        │                      │
        │ messages = [         │
        │   system,            │
        │   user,              │
        │   assistant (think), │
        │   tool (think res),  │
        │   assistant (new),   │
        │   tool (new res),    │
        │   ...                │
        │ ]                    │
        │                      │
        │ response = llm.chat( │
        │   messages,          │
        │   tools              │
        │ )                    │
        │                      │
        │ response.tool_calls  │
        │   = [{               │
        │     name: "preval",  │
        │     args: {...}      │
        │   }]                 │
        │                      │
        └──────────┬───────────┘
                   │ has_tool_calls=true
                   │
        ┌──────────▼──────────────────────┐
        │ Execute: "prevalence"            │
        │                                  │
        │ ctx = _build_ctx(report_dir)    │
        │ result = registry.execute(       │
        │   "prevalence",                  │
        │   {top_n: 20, ...},             │
        │   ctx=ctx                        │
        │ )                                │
        │                                  │
        │ [skill runs: generates report]  │
        │                                  │
        │ messages updated...              │
        │                                  │
        └──────────┬──────────────────────┘
                   │
        ┌──────────▼──────────────────┐
        │ Round 3                      │
        │ (same pattern)               │
        │                              │
        │ ... potentially more tools   │
        │                              │
        │ OR                           │
        │                              │
        │ response.has_tool_calls =    │
        │   false                      │
        │                              │
        │ Final text response ready    │
        │                              │
        └──────────┬──────────────────┘
                   │ has_tool_calls=false
                   │
        ┌──────────▼──────────────────────────┐
        │ Return Final Response                │
        │                                      │
        │ messages.append({                    │
        │   role: "assistant",                 │
        │   content: response.text             │
        │ })                                   │
        │                                      │
        │ _post_run() housekeeping:            │
        │ - Index session in memory            │
        │ - Trigger background review          │
        │                                      │
        │ return response.text                 │
        │                                      │
        └──────────────────────────────────────┘
```

---

## Diagram 8: Config to Runtime Flow

```
┌─────────────────────────────────────────────┐
│           config.py (pydantic)              │
│                                              │
│ custom_skills_dir: Path = Path("./custom")  │
│ (line 87)                                   │
│                                              │
│ Can be overridden via .env:                 │
│ CUSTOM_SKILLS_DIR=/path/to/skills          │
│                                              │
└────────────────┬────────────────────────────┘
                 │
    ┌────────────▼─────────────────────────┐
    │ Settings loaded in __init__           │
    │ get_settings() → Settings(**overrides)│
    │                                        │
    │ settings.custom_skills_dir =          │
    │   Path("./custom_skills")             │
    │   (or from .env if specified)         │
    │                                        │
    └────────────┬────────────────────────┘
                 │
    ┌────────────▼─────────────────────────────────┐
    │ Agent.__init__(settings)                      │
    │ [agent.py:63]                                │
    │                                              │
    │ Pass settings to discovery:                  │
    │                                              │
    │ autodiscover_skills()                       │
    │ # Loads from biobank_agent/skills/          │
    │                                              │
    │ discover_custom_skills(                     │
    │   settings.custom_skills_dir  ◄─────────┐  │
    │ )                                        │  │
    │ # Loads from ./custom_skills/ (or env)  │  │
    │                                          │  │
    │ self.registry = get_registry()          │  │
    │                                          │  │
    └────────────┬──────────────────────────────┤─┘
                 │                              │
    ┌────────────▼──────────────────┐          │
    │ Registry populated with:       │          │
    │ - 44 built-in skills          │          │
    │ - N custom skills             │          │
    │                               │          │
    │ agent.registry.tool_schemas() │          │
    │   → 44+N OpenAI schemas       │          │
    │                               │          │
    │ agent.registry.list_skills()  │          │
    │   → List for /skills command  │          │
    │                               │          │
    │ agent.registry.execute()      │          │
    │   → Execute during run()      │          │
    │                               │          │
    └───────────────────────────────┘          │
                                                │
                                    ./custom_skills/
                                    ├── my_skill.py
                                    └── another.py
```

