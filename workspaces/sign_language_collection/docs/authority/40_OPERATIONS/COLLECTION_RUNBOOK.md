# Semantic Collection Runbook

Status: draft
Owner: Semantic sign-language collection lane
Authority scope: Supported operator sequence for pilot and production sign-language recording sessions.
Last reviewed: 2026-08-12

## Preconditions

Do not book or record a production participant until readiness gates `G0` through `G4` in
[readiness and QC](../50_VALIDATION/READINESS_AND_QC.md) pass. Operators use only the frozen protocol, content pack,
device configuration, calibration procedure, consent package and session manifest template.

## 1. Before The Collection Day

1. Confirm participant eligibility, consent version, appointment and access needs through the restricted registry.
2. Allocate a pseudonymous `participant_id` and planned `session_id`; do not place a name in research paths.
3. Generate the session plan from the frozen core/stress content and condition matrix.
4. Confirm equipment reservation, storage headroom, backup destination, software/firmware versions and clock/trigger
   setup.
5. Verify operator and sign-language reviewer coverage. Do not let participant arrival become the first protocol
   rehearsal.

## 2. Room And Equipment Preflight

1. Record site/environment code and any deviation from the approved room layout.
2. Mount radar/reference devices using the frozen coordinate and placement procedure.
3. Record device serial/config/channel-map identity and run calibration.
4. Run a short empty-scene/background recording and a known motion/sync test.
5. Confirm expected raw byte growth, frame/sample count, reference visibility, timestamps and backup write access.
6. Stop before participant recording if any preflight gate fails; do not compensate by changing unversioned settings.

## 3. Participant Intake

1. Reconfirm consent and the participant's right to pause or withdraw.
2. Explain the recorded modalities and reference video identifiability without overstating radar privacy.
3. Review the session flow, breaks, prompts and re-record policy.
4. Complete only approved metadata and signer eligibility checks.
5. Conduct a short practice using material excluded from the evaluation corpus.

## 4. Core Semantic Recording

For each planned `utterance_id`:

1. Create a new `take_id` before acquisition.
2. Present the frozen prompt and allow the defined preparation interval.
3. Start radar/reference streams and record the synchronization marker.
4. Record the complete continuous utterance without operator-side trimming of raw files.
5. End all streams and run the immediate completeness/sync/signal preview.
6. Ask the language reviewer to mark `accepted`, `re-record`, or `defer` for semantic correctness.
7. If repeated, create a new attempt/take linked to the previous one; never overwrite it.
8. Schedule breaks according to the frozen burden limit and participant request.

## 5. Stress Subset

Only participants assigned by the frozen matrix perform Tier B. Establish a frontal nominal reference for the same
compact utterance subset, then record the specified boundary conditions. At minimum, the plan includes nominal
`30 degree` and `60 degree` off-axis conditions and partial hand or object occlusion. Record actual measured/verified
condition metadata and deviations; a label alone is insufficient.

Do not improvise extra angles, occluders or sentence lists during an individual session. A protocol change must be
versioned and applied consistently to a declared stratum.

## 6. End-Of-Session Review

Before releasing equipment or dismissing the participant when practical:

1. Compare planned versus recorded utterance/condition coverage.
2. Verify every artifact can be read and matches expected sizes/counts.
3. Review automated synchronization and radar/reference health reports.
4. Resolve deferred semantic decisions or schedule targeted re-records.
5. Record deviations, participant interruptions and excluded takes with typed reason codes.
6. Copy the session to the protected backup target and verify checksums.

## 7. Publication And Daily Close

1. The data steward verifies consent scope, pseudonymous metadata, package completeness and checksum inventory.
2. Publish the session atomically to a new immutable versioned destination.
3. Update the collection coverage/QC ledger from the published manifest, not from operator memory or filenames.
4. Review daily failure patterns. Pause production if the same hardware, semantic or operator issue repeats.
5. Record a dated Log for pilot completion, incidents, protocol revisions and production milestones; routine healthy
   sessions need no standalone narrative document.

## 8. Stop Conditions

Immediately pause the affected take or session for consent withdrawal, participant discomfort, wrong participant or
content identity, device/config drift, failed calibration, missing modality, storage pressure, corrupted raw data,
sync failure, unreviewable signing, or any unapproved protocol deviation. Preserve partial artifacts as restricted
incident evidence and never promote them as accepted samples.

Production-wide pause is required when protocol validity, ethics scope, signer eligibility, synchronization or data
loss affects more than an isolated take. Resume only after a documented root-cause decision and any necessary new
protocol version.
