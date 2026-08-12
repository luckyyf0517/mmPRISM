# Project Authority Changelog

Status: current
Owner: mmPRISM coordinator
Authority scope: Material changes to project boundaries, shared contracts, or project Authority.
Last reviewed: 2026-08-12

## 2026-08-12

- Replaced the monolithic revision control plane with project Authority and five business workspaces.
- Kept canonical shared code, configuration, and tests at repository root.
- Established lightweight routine handoff and frozen cross-workspace delivery rules.
- Preserved old documentation paths as compatibility entrypoints and moved dated evidence to Logs.
- Accepted `DEC-039`, superseding `DEC-001` as the current documentation control-plane decision.
- Added a dedicated semantic sign-language collection workspace with a roughly 30-participant target, while
  explicitly excluding legacy non-semantic gestures from semantic cohort and translation evidence.
- Accepted Chinese Sign Language (CSL) as the primary language of the new semantic collection while leaving its
  precise variety/register and written translation target for expert review.
- Simplified the collection around the revision deadline: seek 3--4 professional/proficient CSL contributors when
  available and scale with video-guided volunteers, without maintaining a recruitment funnel.
- Archived CSL-News as checkpoint-side visual-pose evidence and removed its local source/download/cache layer;
  CSL-Daily intake and the new semantic CSL collection are now the active data-rebuild paths.
- Registered the author's incoming historical WaveLLM bundle as preservation-only while transfer is in progress.
  The existing CSL-News-derived mT5-only export remains a fallback until a stable receipt and controlled audit
  establish the incoming checkpoint identity and admissible role.
- Added a lightweight, non-authoritative literature-note area and recorded Uni-Sign's pre-training-scale evidence
  without promoting its full-data result into a revision requirement.
- Relocated the original-submission forensic codebase (root `run_*.py`, `config/`, legacy `src/` modules, and
  legacy shell wrappers) into an explicit read-only `legacy/` directory without content changes (`DEC-047`);
  `src/` now contains only the canonical `mmprism` package.
- Added one cross-workspace research execution model defining cam-pose, synthetic radar, synthetic/real-domain
  mmw-pose, radar features, model roles, stage handoffs, and the CSL-Daily end-to-end control path; workspace
  operation pages retain their own commands, gates, outputs, and current state.
