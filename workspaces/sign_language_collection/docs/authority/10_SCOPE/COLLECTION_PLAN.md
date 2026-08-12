# Revision-Focused CSL Collection Plan

Status: draft
Owner: Semantic sign-language collection lane
Authority scope: Lightweight scope, participant mix, content, recording matrix, phases, and claim boundary for the new CSL collection.
Last reviewed: 2026-08-12

## Goal

Collect a new real radar/reference dataset for the paper revision. Historical self-collected data is non-semantic
gesture material, so the new collection uses fixed Chinese Sign Language (CSL) reference videos with known text
meaning and records participants reproducing those videos.

The target remains approximately 30 participants with usable recordings. We do not maintain contacted, registered,
screened or replacement counts. The final dataset reports only the actual recorded participant count and usable
take count.

## Participant Mix

Use two explicit participant types:

1. `professional_or_proficient_signer`: aim for 3--4 people who know CSL, if they can be found. They can check the
   reference set, demonstrate difficult items and contribute a small high-quality reference subset.
2. `video_guided_volunteer`: the main source of scale. Volunteers watch a fixed CSL video, learn/rehearse it, and
   reproduce it during synchronized radar and reference-video capture.

Professional/proficient signer availability is desirable but is not a gate for starting the pilot. Do not label a
video-guided volunteer as a fluent signer. This dataset can test cross-participant reproduction and sensor/model
robustness; by itself it cannot establish natural CSL use by the Deaf/signing community.

## Minimal Reference Set

Before the pilot, freeze one versioned table containing:

```text
utterance_id
reference_video
Chinese text meaning
core or stress subset
notes for obvious ambiguity/difficulty
```

Prefer meaningful continuous CSL clips from a legally usable, known source such as the project CSL resources. Do
not construct the primary set from arbitrary motions. Keep the set small enough that volunteers can learn it in a
reasonable session. If a professional/proficient contributor is available, ask them to check the selected videos
and meanings; otherwise record that validation is limited to the source dataset/video labels.

The exact clip count, repetition count and session duration are decided by the pilot, not by a large protocol
document.

## Recording Matrix

### Core set

Every participant watches, rehearses and performs the same core reference clips in the nominal frontal setup.
Record raw radar and a synchronized reference video. The reference video is needed to confirm whether the motion
was reproduced and to obtain pose evidence.

### Small revision stress set

Use a small fixed subset, not the full clip set, for:

- nominal frontal recording;
- off-axis `30 degree` and `60 degree` recordings;
- partial hand occlusion;
- object occlusion.

Record both professional/proficient contributors and several volunteers in the stress subset when available. Do
not require all participants to repeat every condition. Hand-to-hand overlap is marked when naturally present;
there is no separate overlap experiment unless later evidence requires it.

## Simple Session Flow

1. Assign an opaque participant ID and record participant type.
2. Confirm the participant agrees to radar and reference-video recording.
3. Check radar/reference capture and synchronization with one trial.
4. Show one frozen CSL reference clip; allow replay and rehearsal.
5. Record the participant's reproduction and immediately check both files.
6. Re-record only when the motion is clearly incomplete or files/synchronization fail.
7. Back up the session and generate checksums.

## Minimal Data Fields

Keep only what the paper and downstream processing need:

- participant ID and participant type;
- session/take/utterance IDs;
- reference-video/content version;
- frontal/orientation/occlusion condition;
- raw radar and reference-video locations/checksums;
- radar configuration/calibration identity;
- synchronization and take status;
- short failure reason when unusable.

No names or contact details belong in the research manifest. Do not add demographic fields unless the paper will
actually analyze them and collection is approved.

## Execution Phases

| Phase | Action | Exit condition |
|---|---|---|
| 1. Reference freeze | select CSL videos and text meanings; seek optional professional check | one versioned core/stress list |
| 2. Setup | prepare minimal consent, radar/reference capture, IDs and backup | one readable synchronized trial package |
| 3. Pilot | record a few volunteers; optionally one proficient signer | workload and basic QC are workable |
| 4. Main collection | record toward approximately 30 participants | usable core coverage and compact stress subset |
| 5. Handoff | freeze manifest, raw files, configuration, QC and checksums | data rebuild accepts the snapshot |

## Paper Claim Boundary

The manuscript must report separately:

- number of professional/proficient CSL contributors actually recorded;
- number of video-guided volunteers;
- source and size of the fixed CSL reference set;
- that volunteers learned/reproduced prompted videos rather than producing spontaneous natural CSL;
- participant-disjoint evaluation and which participants appeared in the stress subset.

This design provides new real semantic-aligned radar data and more performer diversity than the old non-semantic
cohort. It does not justify claiming population-level generalization to fluent sign-language users unless enough
professional/proficient signer evidence is actually collected.

## Completion

Stop expanding the plan once the paper-facing minimum is met: approximately 30 participants with usable core data,
a small verified stress subset, readable synchronized artifacts, a participant-aware manifest and an honest
dataset description. Further linguistic annotation or public release preparation is separate work.
