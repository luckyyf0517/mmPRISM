## ADDED Requirements

### Requirement: Legacy gestures remain outside the CSL dataset

Historical self-collected non-semantic gestures SHALL NOT contribute to the new CSL participant, utterance or
translation totals.

#### Scenario: Report new collection coverage

- **WHEN** collection statistics are generated
- **THEN** they use only the new reference-video-based collection
- **AND** legacy gesture data is reported separately.

### Requirement: The collection preserves participant type

The collection SHALL target approximately 30 participants with usable recordings and SHALL record each as either
`professional_or_proficient_signer` or `video_guided_volunteer`. Professional/proficient contributors SHOULD be
sought with a planning aim of 3--4, but their availability SHALL NOT block volunteer pilot or main collection.
Registration, screening and replacement funnel counts SHALL NOT be required.

#### Scenario: No professional signer is available for the pilot

- **WHEN** the frozen reference set and radar/reference setup are ready but no professional/proficient CSL
  contributor has been found
- **THEN** the team may run the video-guided volunteer pilot
- **AND** the limitation remains explicit in the collection state and later paper text.

### Requirement: Volunteers reproduce frozen CSL reference videos

Every primary volunteer take SHALL bind an `utterance_id`, frozen CSL reference video and known Chinese text
meaning. The participant SHALL be allowed to replay and rehearse before recording. Basic QC SHALL verify visible
completion against the reference, but SHALL NOT certify the volunteer as a fluent or natural CSL signer.

#### Scenario: Record a volunteer take

- **WHEN** a video-guided volunteer reproduces a reference clip
- **THEN** the take stores the reference-content version and participant type
- **AND** downstream reporting describes it as prompted/video-guided reproduction.

### Requirement: Acquisition uses a common core and compact stress subset

Recorded participants SHALL complete a common core set in the nominal setup. A compact subset SHALL cover frontal,
30-degree, 60-degree, partial hand occlusion and object occlusion conditions without requiring the complete corpus
under every condition.

#### Scenario: Run the stress subset

- **WHEN** a participant is selected for stress recording
- **THEN** only the frozen stress clips and conditions are used
- **AND** operators do not expand the matrix during the session.

### Requirement: Accepted takes preserve minimum evidence

Each accepted take SHALL bind opaque participant/session/take IDs, participant type, reference content, raw radar,
reference video, condition, radar configuration, synchronization result, basic QC state and file checksums. A retry
SHALL use a new take identity instead of overwriting an earlier file.

#### Scenario: A file or reproduction is unusable

- **WHEN** radar/reference data is unreadable, synchronization is unusable or the visible reproduction is clearly
  incomplete
- **THEN** the take is marked rejected/rerecord with a short reason
- **AND** any retry receives a new `take_id`.

### Requirement: Research manifests support participant-disjoint splits

Research data SHALL use opaque participant IDs and explicit relationships. Direct identifiers SHALL remain outside
the research manifest. All takes from one participant SHALL be groupable into one downstream split.

#### Scenario: Build a cross-participant split

- **WHEN** data rebuild assigns train, validation and test data
- **THEN** one participant appears in only one split
- **AND** both participant types remain measurable in the split summary.

### Requirement: Paper claims match the collected cohort

The dataset and manuscript SHALL report professional/proficient CSL contributors and video-guided volunteers
separately. Video-guided volunteer results SHALL NOT be described as spontaneous natural CSL or fluent-signer
population generalization.

#### Scenario: Main collection contains only volunteers

- **WHEN** no professional/proficient CSL participant is ultimately recorded
- **THEN** the paper states this limitation
- **AND** conclusions are restricted to prompted CSL reproduction across participants.
