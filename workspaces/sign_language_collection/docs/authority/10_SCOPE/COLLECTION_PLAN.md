# New Semantic Sign-Language Collection Plan

Status: draft
Owner: Semantic sign-language collection lane
Authority scope: Scientific scope, cohort, content, condition matrix, phases, resources, risks, and success criteria for the new collection.
Last reviewed: 2026-08-12

## Objective

Build a new real-world dataset that can support both radar hand reconstruction and semantic continuous
sign-language translation. The collection must correct the central limitation of the historical self-collected
data: those recordings are non-semantic gestures and cannot establish sentence-level sign-language understanding.

The primary signing language is **Chinese Sign Language (CSL)**. Here CSL means an actual sign language used by
eligible signers, not arbitrary gestures, signed spoken-Chinese word order, or a prompt-by-prompt hand-motion
encoding. The exact regional/register scope and written translation target remain to be frozen with the
sign-language lead.

The production target is approximately 30 **usable** participants. The final paper reports the actual accepted
count, not the recruitment target or number of attempted sessions. The historical 12-person cohort is reported
separately and is never added to the new cohort count.

## Scientific Questions

The new dataset is designed to answer the following without claiming that every protocol choice was dictated by
the reviewers:

1. Can a model trained with synthetic/visual supervision operate on real semantic sign-language radar data?
2. Does performance generalize to unseen signers with different hand sizes and signing styles?
3. How do reconstruction and translation behave away from frontal alignment and under partial hand/object
   occlusion, including the reviewer-cited `30 degree` and `60 degree` conditions?
4. What semantic scope, vocabulary, sentence distribution and non-manual dependencies does the dataset actually
   cover?

## Dataset Boundary

### Included as primary evidence

- Meaningful continuous sign-language utterances from eligible signers.
- Synchronized raw radar, reference recording sufficient for pose/semantic verification, and canonical text
  meaning for each accepted take.
- Participant, session, environment, orientation and occlusion metadata stored under pseudonymous identities.
- Explicit invalid/re-record decisions and QC evidence.

### Included only as calibration or secondary analysis

- Static poses, isolated signs and non-semantic motion used for sensor checks or pose calibration.
- Pilot recordings, unless formally promoted after proving protocol identity.
- Naturally occurring hand-to-hand overlap labels and failure cases.

### Excluded from the new semantic cohort

- All historical self-collected non-semantic gesture recordings.
- Attempts from participants who do not pass the frozen signer eligibility rule.
- Takes without verified meaning, usable radar/reference synchronization, consent coverage or session QC.
- Synthetic radar and public visual corpora; these remain separate upstream datasets.

## Participant Plan

### Target

- Planning target: approximately 30 accepted participants.
- Counting unit: one pseudonymous participant with at least one fully accepted core session.
- Replacement rule: an ineligible participant or failed session does not count toward the target; the failure is
  retained in the restricted recruitment/session ledger and a replacement may be recruited.
- Downstream rule: participant identity is the minimum split group. Frame- or take-random splits cannot support
  new-user generalization.

### Eligibility to freeze before recruitment

- Default to adults unless ethics approval explicitly covers another population.
- Ability to perform the frozen CSL variety/register at the required level.
- Eligibility assessed by a named sign-language reviewer role using a recorded rubric, not by self-report alone.
- Ability and consent to complete radar and reference recordings and the planned session duration.
- Exclusion/withdrawal criteria and compensation treatment defined before the first contact.

Deaf/hearing status, native/second-language status, age band, handedness, hand measurements and other attributes
are collected only when scientifically justified, approved and consented. They are used to characterize coverage,
not as informal proxies for sign-language competence.

### Recruitment strategy

Recruit through institutionally approved university, community, interpreter/education and collaborator channels.
Use one screening form and one eligibility decision process across channels. Recruitment messaging must state the
recorded modalities, identifiability of reference video, storage/release plan, compensation, withdrawal boundary
and session burden. No production booking occurs before ethics and consent approval.

## Semantic Content Plan

The content pack must be versioned and frozen before the pilot. It contains:

```text
content_pack_id and version
sign language: Chinese Sign Language (CSL)
CSL regional/register scope
utterance_id
canonical written translation/meaning and target language
allowed paraphrases, if any
prompt shown to the participant
content category and lexical tags
expected continuous/isolated form
non-manual dependency: required / optional / none / unknown
known ambiguity or exclusion note
sign-language reviewer and review status
```

The primary set consists of meaningful continuous utterances rather than concatenated arbitrary gestures. The
pack should cover varied sentence lengths and everyday semantic categories without claiming open-vocabulary
coverage beyond its actual inventory. Items that require facial/body grammar must either be captured and annotated
with an appropriate reference modality or explicitly marked outside the hand/radar-only task boundary.

The primary language is fixed as CSL. Its precise variety/register, written translation target, vocabulary,
sentence count, repetition count and length distribution remain open decisions. They are fixed only after a
language expert review and a timed pilot establish correctness and participant burden.

## Efficient Two-Tier Acquisition

### Tier A: core semantic corpus

Every accepted participant completes the same frozen core protocol in the nominal calibrated setup. Tier A is the
only component used to count progress toward approximately 30 usable participants. It provides participant-held-out
semantic and pose evidence.

### Tier B: compact real-world stress subset

A predefined compact subset of utterances is repeated under selected boundary conditions. At minimum the plan
evaluates the reviewer-cited off-axis `30 degree` and `60 degree` conditions and partial hand or object occlusion,
with reconstruction and translation both retained as downstream targets. A frontal nominal take anchors each
stress comparison.

The participant count and utterance count for Tier B are not yet fixed. They must cover multiple held-out signers
and be selected before production analysis, but should not force all 30 participants to repeat the full semantic
corpus under every condition. Hand-to-hand overlap is labeled when present; a separate forced-overlap condition is
added only if the mechanism explanation cannot be supported by existing Tier A/B examples.

## Participant-Disjoint Evaluation Design

Before production collection, define a split policy over pseudonymous participant IDs. A provisional planning
allocation for 30 accepted participants is `20 train / 5 validation / 5 test`; this is not final until statistical,
recruitment and adaptation needs are reviewed. The final assignment must be frozen by `data_rebuild`, contain no
participant overlap, and preserve a genuinely held-out test cohort.

Any calibration or few-shot adaptation study records exactly how much held-out-participant data was exposed. A
participant cannot simultaneously contribute calibration data and be described as zero-shot. Stress-condition
results must retain condition labels rather than being collapsed into one overall average.

## Phases And Deliverables

| Phase | Main work | Deliverable |
|---|---|---|
| 0. Freeze scientific scope | CSL variety/register, written target, task boundary, cohort target, core/stress principles | approved protocol version |
| 1. Ethics and recruitment | approval, consent, compensation, screening, identity separation | approved participant package |
| 2. Content and hardware | reviewed content pack, radar/reference config, synchronization, calibration | immutable protocol bundle |
| 3. Pilot | small eligible cohort, full operator rehearsal, timing and QC | dated pilot report and go/no-go decision |
| 4. Production | core collection for approximately 30 accepted participants plus compact stress subset | immutable raw session packages and live QC ledger |
| 5. Semantic and technical QC | sign correctness, target text, signal/sync/completeness checks, re-records | accepted/rejected take inventory |
| 6. Freeze and handoff | pseudonymous manifest, checksums, protocol/config copies, coverage summary | frozen collection snapshot accepted by data rebuild |

## Roles

- **Collection lead:** owns protocol version, schedule, operator training and stop decisions.
- **Sign-language lead/reviewer:** owns eligibility rubric, content correctness, semantic review and ambiguity policy.
- **Radar lead:** owns radar configuration, array/channel map, calibration and raw-signal health checks.
- **Reference/pose lead:** owns camera/reference setup, synchronization and pose-ground-truth suitability.
- **Data steward:** owns consent scope, identity separation, access control, immutable session publication and checksums.
- **QC reviewer:** independently accepts/rejects sessions and records reasons; the operator cannot silently waive a gate.

One person may hold multiple roles, but every production session records the responsible people by role.

## Major Risks And Controls

| Risk | Control |
|---|---|
| Recruiting approximately 30 competent signers is slow | freeze eligibility early, recruit through multiple approved channels, pilot scheduling burden, request revision extension before evidence quality is compromised |
| Participants reproduce prompts incorrectly or unnaturally | language-reviewed content, rehearsal, per-take semantic review, explicit re-record reasons |
| Full condition matrix becomes infeasible | core-for-all plus compact stress subset; freeze a minimum matrix before production |
| Radar/reference streams drift or silently drop | hardware bench test, sync markers, per-session automated integrity report and immediate backup |
| Reference video creates identifiable data | explicit consent, restricted identity map, access tiers and release-specific redaction/derived products |
| Dataset leakage inflates new-user results | subject identity in manifest, participant-disjoint split, immutable test assignment and exposure ledger |
| Protocol changes halfway through production | versioned protocol; material changes trigger a new cohort/protocol stratum and cannot be silently pooled |

## Completion Criteria

Collection is complete only when:

- approximately 30 usable participants have accepted core sessions under the frozen protocol;
- every accepted take has verified semantics, required modalities, synchronization, metadata and checksums;
- stress-subset coverage matches the frozen matrix or deviations are explicitly recorded;
- invalid and missing takes remain visible in the QC ledger;
- participant identity is pseudonymous in research manifests and direct identifiers remain separately restricted;
- the final manifest, protocol, content pack, calibration/configuration, QC summary and hashes are immutable;
- `data_rebuild` accepts the handoff and independently validates sample/participant coverage.
