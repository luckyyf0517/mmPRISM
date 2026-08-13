# Lightweight Collection Readiness And QC

Status: current
Owner: Semantic sign-language collection lane
Authority scope: Minimum go/no-go and take acceptance checks for the paper-revision CSL collection.
Last reviewed: 2026-08-12

## Readiness

| Gate | Requirement | State |
|---|---|---|
| `G0` scope | CSL, approximately 30 total participants, mixed participant types and legacy exclusion decided | passed |
| `G1` reference set | core/stress CSL videos and Chinese meanings frozen; professional check best-effort | in_progress |
| `G2` setup | minimal consent, IDs, radar/reference configuration, sync and backup tested | blocked |
| `G3` pilot | a few video-guided participants complete a workable session | blocked |
| `G4` main collection | approximately 30 participants have usable core recordings and stress coverage is sufficient | blocked |
| `G5` handoff | manifest/files/checksums validate and data rebuild accepts them | blocked |

Professional/proficient CSL participants are desirable but not a readiness gate because availability is uncertain.

## Take QC

A take is usable when:

- radar and reference video exist, open and match the take ID;
- the visible reproduction is complete enough to compare with the fixed reference clip;
- synchronization is usable for radar/reference alignment;
- participant type, utterance and condition are recorded;
- files are backed up and checksummed.

There is no multi-stage review workflow. Use `accepted`, `rejected` or `rerecord`, plus a short failure reason.

## Coverage

Generate only the statistics needed for the paper and split:

- total recorded participants with usable core data;
- professional/proficient signer count and video-guided volunteer count;
- accepted take count by reference clip and condition;
- core/stress missing and rejection counts;
- participant-disjoint split coverage.

Do not maintain contacted, registered, screened or consented-only funnel counts. Do not merge historical
non-semantic gestures into these totals.

## Claim Check

Before handoff, verify that dataset text reports participant types accurately (`video_guided_volunteer` versus
`professional_or_proficient_signer`) and that the volunteer training protocol is described. If no
professional/proficient contributors are found, record that directly in the dataset facts. Paper-facing claim
wording follows these disclosed facts and is finalized during manuscript writing (`DEC-055`).
