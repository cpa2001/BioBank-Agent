# Changelog

All notable changes to Biobank Agent are documented here.

## [2.0.0] — 2026-04-22

### Added
- **Scientific Discovery Pipeline**: `discover`, `predict`, `gwas_proxy` skills for automated biomarker discovery
- **Literature Research**: `web_search`, `web_fetch`, `read_pdf`, `fetch_paper`, `read_paper`, `deep_research` skills
- **Scientific Writing**: `nature_writer` for Nature-quality manuscript sections
- **Research Ideation**: `brainstorm`, `critical_thinking` skills for hypothesis generation
- **Plan Mode**: Structured planning workflow (INTAKE → ALIGNMENT → EXECUTION → DONE)
- **Smart Plotting**: `smart_plot` with Nature/ICML/NEJM/Lancet style presets
- **CLI Commands**: 20 slash commands (`/plan`, `/compact`, `/clear`, `/cost`, `/export`, `/help`, etc.)
- **Token Tracking**: Cumulative prompt/completion token counting
- **Hot-Reload Skills**: `custom_skills/` auto-discovery, in-session skill activation
- **Report Dual Format**: Technical report (`format='report'`) and IMRaD paper draft (`format='paper'`)
- **SVG/PDF Figures**: Publication-quality vector output as default
- **ICML Figure Style**: `icml_figure()` for ML conference formatting
- **Panel Labels**: `add_panel_labels()` for multi-panel figures
- New dependencies: pymupdf, httpx, html2text, beautifulsoup4, duckduckgo-search

### Changed
- Renamed CLI entry point to `biobank` (legacy alias removed)
- Report system rewritten with interpretive text and Key Findings
- Plotting defaults to SVG+PDF instead of PNG+PDF
- `create_skill` now auto-activates via `custom_skills/` hot-reload
- Enhanced `registry.py` with `reload_skill()`, `unregister()`, `discover_custom_skills()`

### Fixed
- Correlation skill now uses `save_figure()` instead of manual `savefig()`

## [1.0.0] — 2026-04-21

### Added
- Initial release: 28 skills, DuckDB data layer, ReAct agent loop
- Parquet rebuild from 484 to 4,971 field IDs
- LLM client with retry logic and exponential backoff
- Self-evolution: error catalog, pipeline recording, skill generation
- Nature-style plotting (300 DPI, Okabe-Ito palette)
- 4-tier memory system
- 164 unit tests
