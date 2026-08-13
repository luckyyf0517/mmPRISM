# Reference Set And Scale Plan (Early Draft)

Status: draft
Owner: Semantic sign-language collection lane
Authority scope: Early-stage textual planning for reference content, screening, scale and time estimates.
Last reviewed: 2026-08-13

This page is earlier and softer than [COLLECTION_PLAN.md](COLLECTION_PLAN.md). Nothing here freezes protocol;
the pilot and the external collection-system build decide the real numbers.

## Reference Frame

The reference content frame is the full CSL-Daily semantic inventory, measured on the received
direct-preservation upload (`external/csl_daily/csl_daily_original_20260812`):

| Quantity | Value |
|---|---|
| Total sequences (videos) | 20,654 |
| Unique semantic episodes (sentence IDs) | 7,398 |
| Signers | 10 |
| Renditions per episode | mean 2.79 (1x: 50, 2x: 1,650, 3x: 5,488, 4x: 210) |
| Episode duration (500-sequence sample, 30 fps) | mean 4.0 s, p50 3.7 s, p90 6.2 s, max 13.0 s; 99.2% within 10 s |

Using CSL-Daily as the reference frame keeps the real collection semantically aligned with the synthetic
rebuild chain: the same sentence can carry a simulated radar cube, a real radar recording, and the official
Chinese text, which is exactly the matched sim2real comparison the revision needs.

## Screening Round

The 7,398 episodes are a frame, not a usable list. Before any reference freeze, run one screening pass:

1. Duration: keep episodes at or below the ~10 s recording budget (99.2% pass by measurement).
2. Rendition choice: prefer episodes with 2-3 renditions so the clearest reference video can be selected;
   the 50 single-rendition episodes are last-resort candidates.
3. Reference-video quality: single signer fully in frame, both arms visible, no distracting background
   motion; reject ambiguous or error-containing renditions.
4. Content: Chinese text from the official annotation must be complete and appropriate for prompted
   reproduction; flag culturally or linguistically ambiguous items for the professional check.
5. Overlap with the synthetic chain: prefer episodes covered by the rebuilt CSL-Daily annotation/simulation
   lineage so real and synthetic evidence pair one-to-one.
6. Professional/proficient signer check when available (outreach in progress); otherwise record that
   validation is limited to source labels.

The screening output is one versioned table (`utterance_id`, chosen reference video, Chinese text,
core/stress assignment, difficulty notes) — the same artifact COLLECTION_PLAN requires before the pilot.

## Scale Scenarios (single device, episode <= 10 s)

Per-take cycle for a video-guided volunteer is estimated at 2-4 minutes (watch 15-30 s, rehearse 1-3 min,
record ~10 s, immediate check 10-20 s). Proficient signers run 3-4x faster. A 30-episode session is
about 2.5 hours including consent, synchronization trial and breaks.

Semantic coverage is an open planning parameter, not capped by this plan. The 30-participant frame combines
one shared core set (cross-participant comparison and the stress conditions) with per-participant expansion
sets drawn disjointly from the 7,398-episode frame, so total coverage scales with the measured per-take cycle:

| Expansion per participant | Total distinct episodes (30 participants) | Coverage of 7,398 |
|---|---|---|
| 30 (shared core only) | ~30 | ~0.4% |
| 100 | ~3,000 | ~40% |
| 250 | ~7,398 | effectively full frame |

Multiple sessions per participant are expected for larger expansion targets. Final scale is decided only
after the collection system is built and the real per-take cycle is measured in the pilot.

## Timeline Estimate

```text
G1 reference freeze (screening round above)   3-5 days ┐
G2 setup (consent, capture/synchronization)   ~1 week  ┘ parallel
G3 pilot (2-3 volunteers)                     2-3 days
G4 main collection (~30 participants, 1 rig)  2-3 weeks
G5 freeze and handoff                         2-3 days
```

Total roughly 4-6 weeks after kickoff. Budget order of magnitude: volunteers ~200 CNY per 2.5 h session
(~6k total), professional/proficient contributors 600 CNY/day (2-5k total); under 10k CNY overall.

## Open Questions

- Legality/permission boundary for using CSL-Daily videos as on-site reference material.
- CSL regional/register boundary and written translation target (expert review pending).
- Professional/proficient signer availability (social-media outreach in progress).
- Whether a second radar rig or a learn-while-others-record pipeline can raise single-device throughput.
