# Reviewer Release Audit

Status: `automated_inventory_active_release_blocked`
Last Updated: `2026-08-11`

## Purpose

The reviewer archive must be assembled from an explicit allowlist, not by copying the development
repository. The repository intentionally retains legacy code and internal revision material for evidence
recovery, while the public archive must contain only canonical, tested entry points.

`mmprism release-audit` validates this boundary without creating or deleting an archive. Its versioned
profile is `configs/release/reviewer_release_v1.yaml`.

## Selection Contract

The auditor starts from `git ls-files`, applies configured include and exclude rules, and then:

1. requires every declared release deliverable to exist and be selected;
2. rejects selected legacy, internal, credential, manuscript, and agent-only paths;
3. rejects symlinks and tracked paths that are no longer regular files;
4. hashes every selected file with SHA-256 and records its byte size;
5. scans UTF-8 content for configured local-path and credential patterns;
6. validates `[project.scripts]` against the expected console entrypoints;
7. parses canonical Python with `ast` to identify missing internal modules, relative imports, legacy
   namespace imports, external packages, dependency edges, and strongly connected components;
8. records the exact Git commit/state and a fingerprint of the resolved audit configuration.

Formal release evidence requires a clean Git tree. The JSON report is written atomically and returns exit
code `0` only when no finding remains; a valid report with blockers returns `1`, while malformed config or
execution errors return `2`.

## Development/Release Boundary

The following tracked material remains available in the development repository but is forbidden from the
reviewer selection:

- `CLAUDE.md`, `AGENTS.md`, `.env`, and `.gitmodules`;
- `paper/manager/` and the private `paper/manuscript` submodule;
- root `run_*.py`, legacy `config/`, `requirements.txt`, and legacy `src/*` namespaces;
- historical model shell scripts that still invoke unsupported legacy entrypoints;
- the private Overleaf helper and operational host configs.

This is an exclusion boundary, not a deletion policy. Legacy material stays read-only until the original
submission evidence audit is complete.

## Current Blocking Deliverables

The reviewer profile deliberately requires artifacts that are not yet available, so it cannot silently
become a misleading partial archive:

- an author-approved `LICENSE`;
- a runnable portable radar example;
- a runnable portable mT5 example.

Phi-3 is not supported. Its legacy implementation and claims remain excluded, and the release content
gate rejects reintroduction of its name into the selected public surface. The full re-entry requirements
are documented in `model_support.md`.

## Command

```bash
uv run mmprism release-audit configs/release/reviewer_release_v1.yaml \
  --output paper/manager/evidence/artifacts/release_audit_v1.json
```

The output is an engineering and reviewer-response artifact. It is not itself included in the public
release inventory.
