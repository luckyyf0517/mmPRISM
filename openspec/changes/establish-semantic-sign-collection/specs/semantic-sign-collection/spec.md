## ADDED Requirements

### Requirement: Semantic collection is distinct from legacy gestures

The system SHALL treat all historical self-collected project recordings as non-semantic gesture evidence unless a
separate verified semantic source exists. Historical recordings SHALL NOT contribute to the participant,
utterance, sentence, or translation totals of the new semantic sign-language dataset.

#### Scenario: Build semantic corpus coverage

- **WHEN** the collection coverage report counts accepted semantic participants and utterances
- **THEN** it selects accepted takes from the new semantic collection protocol only
- **AND** legacy self-collected gesture records are reported separately and excluded from semantic totals.

### Requirement: Collection progress counts usable participants

The collection SHALL target approximately 30 usable participants. A participant SHALL count toward this target
only after passing the frozen eligibility rule and receiving acceptance for at least one complete core semantic
session. Attempted, contacted, consented-only, pilot-only, withdrawn, ineligible, or technically rejected sessions
SHALL NOT be counted as usable production participants.

#### Scenario: A recruited session fails quality control

- **WHEN** a recruited participant's only production session lacks valid synchronization or semantic review
- **THEN** the session remains visible with its failure reason
- **AND** the participant is not included in the usable-participant count.

### Requirement: Primary records contain verified semantic signing

The primary dataset SHALL record Chinese Sign Language (CSL) and SHALL consist of meaningful continuous CSL
utterances bound to a frozen CSL variety/register, content-pack identity and canonical meaning. A qualified
sign-language reviewer SHALL verify performed meaning or an allowed paraphrase; the prompt shown to the participant
SHALL NOT by itself establish semantic ground truth. Arbitrary non-semantic gestures or manually encoded
spoken-Chinese word sequences SHALL NOT be accepted as CSL evidence.

#### Scenario: Performed signing differs from the prompt

- **WHEN** a participant performs an utterance whose meaning does not match the prompt or an allowed paraphrase
- **THEN** the take is rejected or requested for re-record according to the semantic rubric
- **AND** the prompt text is not silently retained as its target label.

### Requirement: Acquisition uses core and stress tiers

Every usable production participant SHALL complete the frozen core semantic protocol in the nominal calibrated
setup. A frozen compact subset SHALL cover selected real-world boundary conditions rather than requiring every
participant to repeat the entire corpus under every condition. The boundary plan SHALL include off-axis conditions
corresponding to reviewer-cited 30-degree and 60-degree examples and partial hand or object occlusion, retaining
both reconstruction and translation as downstream evaluation targets.

#### Scenario: Schedule boundary-condition collection

- **WHEN** a participant is assigned to the stress tier
- **THEN** the session plan selects only the frozen compact utterance/condition matrix
- **AND** operators do not improvise additional conditions or repeat the complete core corpus.

### Requirement: Participant identity is pseudonymous and split-capable

Research manifests SHALL use stable opaque participant, session, utterance and take identifiers. Direct identity
and consent records SHALL remain in a separate restricted registry. Each accepted take SHALL expose participant
identity as a grouping key for downstream participant-disjoint split construction without exposing direct
identifiers.

#### Scenario: Prepare a new-user evaluation split

- **WHEN** data rebuild constructs train, validation and test assignments from the frozen collection manifest
- **THEN** all takes from one participant are assigned to one split only
- **AND** filenames or direct participant identifiers are not used to infer the relationship.

### Requirement: Accepted takes bind raw acquisition and quality evidence

Every accepted semantic take SHALL bind immutable raw radar, synchronized reference evidence, exact acquisition
configuration, calibration, synchronization diagnostics, semantic identity, typed condition metadata, consent
scope, QC state, relative artifact locations and cryptographic checksums. A re-record SHALL receive a new take
identity and SHALL NOT overwrite an earlier attempt.

#### Scenario: A take is re-recorded

- **WHEN** semantic or technical review requests another attempt
- **THEN** the operator creates a new take linked to the prior attempt
- **AND** the rejected or superseded attempt and its QC reason remain immutable.

### Requirement: Production is gated by approval and pilot evidence

Production collection SHALL NOT start until language/content, ethics/recruitment, acquisition/synchronization,
operator dry-run and eligible-signer pilot gates have passed with named evidence. Material content, sensor,
synchronization or consent changes after the pilot SHALL create a new protocol identity rather than silently
pooling incompatible sessions.

#### Scenario: Hardware changes after the pilot

- **WHEN** a material radar configuration or synchronization path changes after the pilot
- **THEN** production remains blocked until the change is characterized and assigned a new protocol identity
- **AND** earlier sessions are not silently treated as protocol-equivalent.

### Requirement: Collection handoff is frozen and independently validated

The collection workspace SHALL publish immutable session packages, a pseudonymous manifest, protocol/content/config
and calibration identities, participant/session/take coverage, QC and exclusion ledger, consent-scope boundary,
producer commit, checksums and validation status. Data rebuild SHALL independently validate this identity before
creating derived tensors or subject-disjoint splits.

#### Scenario: Data rebuild receives the collection

- **WHEN** the collection workspace offers a final snapshot for downstream processing
- **THEN** data rebuild verifies checksums, identity uniqueness, required modalities, participant metadata coverage
  and accepted QC states
- **AND** failure leaves the handoff blocked without modifying raw session packages.
