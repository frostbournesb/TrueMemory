# Changelog

## [0.2.0] - 2026-04-03

### Added
- 9 data visualizations (hero banner, leaderboard bar chart, accuracy vs cost scatter, cost per answer, category radar, latency, hardware matrix, eval pipeline diagram, per-category grouped bars)
- `assets/charts/` directory with chart HTML sources and rendered PNGs
- `benchmarks/` directory with full LoCoMo evaluation against 8 memory systems
- Independent benchmark scripts for each competitor (self-contained, reproducible on Modal)
- Complete result JSONs with per-question answers, judge votes, and latency data
- BENCHMARK_RESULTS.md with cost analysis, latency comparison, and hardware requirements
- LICENSE file (Apache 2.0)
- CHANGELOG.md

### Changed
- Visual README overhaul: hero banner, emoji section headers, highlight badges, embedded charts
- License changed from MIT to Apache 2.0
- Updated README benchmark section: 8 competitors (was 4), best scores across runs
- TrueMemory Pro: 91.5% on LoCoMo
- TrueMemory Base: 88.2% on LoCoMo

### Benchmark Results
- 8 systems evaluated on LoCoMo (1,540 questions each, 12,320 total) with identical answer model, judge, scoring, top-k, and prompt
- TrueMemory Pro: 91.5%, TrueMemory Base: 88.2%
- All runs completed with zero API errors

## [0.1.3] - 2026-03-28

### Added
- TRUEMEMORY_EMBED_MODEL environment variable for tier selection
- GPU optional dependency (`pip install truememory[gpu]`)

## [0.1.2] - 2026-03-27

### Added
- Incremental entity profile building for MCP/add() workflow

## [0.1.1] - 2026-03-26

### Added
- Initial release of truememory
- 6-layer memory pipeline: FTS5, vector search, temporal, salience, personality, consolidation
- Base tier (Model2Vec) and Pro tier (Qwen3) embedding support
- MCP server for Claude integration
- Simple Memory API (Mem0-compatible interface)
