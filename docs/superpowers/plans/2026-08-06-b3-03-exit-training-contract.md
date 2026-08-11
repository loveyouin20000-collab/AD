# B3-03 Exit Policy Training Contract / No-positive Target Handling Plan

Goal: define the training contract for early-exit policy training and handle
the B3-02 no-positive-exit target distribution before any training is allowed.

Decision:

- If all early depths have zero positive exit targets, do not unlock training.
- Emit a conservative full-depth fallback contract.
- Keep depth 24 as mandatory fallback.
- Do not generate an exit-policy checkpoint.

Inputs:

- `docs/phase_b/b3_02_exit_prerequisite_materialization_manifest.json`
- B3 accepted early-exit config

Outputs:

- `docs/phase_b/b3_03_exit_policy_training_contract.json`
- `docs/phase_b/b3_03_exit_policy_training_contract.md`
- `docs/phase_b/b3_03_exit_policy_training_contract_evidence.json`
- SHA sidecars

Boundary:

- training_started = false
- training_unlocked = false when no positive labels
- evaluation_started = false
- final_content_accessed = false
- checkpoint_generated = false
- tracked `.pt` = 0
- not pushed
