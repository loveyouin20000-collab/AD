# B3-02 Exit Target / Latency / Calibration Trace Materialization Plan

Goal: materialize the three prerequisites required by B3-01 early-exit
preflight without training or evaluating an exit policy.

Inputs:

- B3 accepted LSE config
- B2-06D LSE calibration predictions
- B2-06D first controlled run evidence
- B2-06F accepted LSE manifest
- B2-07 phase final closure manifest

Outputs:

- `artifacts/targets/early_exit/b3_02_exit_targets_manifest.json`
- `artifacts/profiles/early_exit/b3_02_latency_profile.json`
- `artifacts/traces/early_exit/b3_02_calibration_trace.jsonl`
- tracked B3-02 evidence docs and SHA sidecars

Boundary:

- training_started = false
- evaluation_started = false
- final_content_accessed = false
- checkpoint_generated = false
- tracked `.pt` = 0
- not pushed
