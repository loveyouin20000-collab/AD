# B2-05C0 Target-Conflict Diagnosis

## Status

- Diagnosis only: **no retraining**, no checkpoint mutation, no accepted manifest
- Qualification unchanged: `localized_but_target_fidelity_unqualified`
- Deployment qualified: `False`
- Classification: **E**
- Single-head fixed-family weighting feasible on calibration: **True**

## Signed-diagnostics boundary correction

- Deployment weights are **not** valid signed-Shapley proxies
- Deployment signed metrics: `not_available_in_deployment_artifact`
- Canonical signed diagnostics bind to `canonical_best_training_checkpoint` with `not_part_of_deployment_artifact=true`
- Deployment export matches best training trunk: `True`

## Frozen identities (unchanged)

| Identity | SHA-256 |
|----------|---------|
| accepted training plan | `59e20f4cb337ef42384f70bb8b3dad5211d906341b0a2d41f7e6847610635980` |
| seed collection | `94a6a9332a0694889c7a0255814ac13fe8316c601529197063165ce14ec1277f` |
| canonical selection | `e3bc06dfa02d6109544648020680d907bf0fce5ed7a093372d74009f9e69e142` |
| deployment scientific | `4cbc6fb88f39ed86deacfbbe48580f7682453b94becb046ec6ef1b1302df378a` |
| evaluation unlock | `19dca41e9f647d12afce9877a7340f5af58bf9a23997d7339dded26d89fe73dd` |
| qualification scientific | `da51e5fc1302cf507bc844f87e82cb66f7d2fa0a13e61f28a0dba14333201c49` |

## Evaluation depth-24 trained model (canonical seed 17)

| Family | DLCM KL macro | Uniform KL macro | delta (DLCM−uniform) |
|--------|---------------|------------------|----------------------|
| GT | 0.109855 | 0.159958 | -0.050103 |
| Teacher | 0.069166 | 0.052692 | 0.016474 |

Per-category teacher KL (eval d24): `{'bottle': 0.041638061584012044, 'carpet': 0.09669301700914863}`
Per-category GT KL (eval d24): `{'bottle': 0.18438878776807974, 'carpet': 0.03532076751791963}`

## Equal-family oracle (most important)

| Split | GT KL macro | Teacher KL macro | both gates pass |
|-------|-------------|------------------|-----------------|
| calibration | 0.102863 | 0.066429 | False |
| evaluation | 0.089428 | 0.065389 | False |

## Alpha feasibility

- Calibration feasible alpha interval: `[0.31, 0.38]`
- Evaluation post-hoc feasible alpha interval (diagnostic only): `[]`
- Evaluation **not** used to choose alpha

## Seed-by-seed evaluation (sealed JSON)

{
  "17": {
    "delta_gt": -0.05010309053824226,
    "delta_teacher": 0.01647361977783915,
    "kl_gt_macro": 0.109854787712021,
    "kl_gt_uniform_macro": 0.15995787825026325,
    "kl_teacher_macro": 0.06916554513914097,
    "kl_teacher_uniform_macro": 0.05269192536130182,
    "per_category_gt": {
      "bottle": 0.1843887852246354,
      "carpet": 0.0353207901994066
    },
    "per_category_teacher": {
      "bottle": 0.04163805168263365,
      "carpet": 0.0966930385956483
    },
    "source": "sealed_evaluation_json",
    "teacher_fails_macro": true
  },
  "29": {
    "delta_gt": -0.047046075664253076,
    "delta_teacher": 0.030985894857961868,
    "kl_gt_macro": 0.11291180258601018,
    "kl_gt_uniform_macro": 0.15995787825026325,
    "kl_teacher_macro": 0.08367782021926369,
    "kl_teacher_uniform_macro": 0.05269192536130182,
    "per_category_gt": {
      "bottle": 0.18136203100488132,
      "carpet": 0.04446157416713902
    },
    "per_category_teacher": {
      "bottle": 0.0551508021012932,
      "carpet": 0.11220483833723419
    },
    "source": "sealed_evaluation_json",
    "teacher_fails_macro": true
  },
  "43": {
    "delta_gt": -0.04824502761268065,
    "delta_teacher": 0.03682080501073115,
    "kl_gt_macro": 0.1117128506375826,
    "kl_gt_uniform_macro": 0.15995787825026325,
    "kl_teacher_macro": 0.08951273037203297,
    "kl_teacher_uniform_macro": 0.05269192536130182,
    "per_category_gt": {
      "bottle": 0.17595842123195995,
      "carpet": 0.04746728004320525
    },
    "per_category_teacher": {
      "bottle": 0.05612787981111033,
      "carpet": 0.12289758093295561
    },
    "source": "sealed_evaluation_json",
    "teacher_fails_macro": true
  }
}

## Selector diagnostic (calibration, read-only)

{
  "constrained_selector_would_select": null,
  "frozen_selector_canonical_seed": 17,
  "mean_family_kl_would_select": 17,
  "note": "diagnostic only; canonical seed unchanged",
  "per_seed_calibration_scores": {
    "17": {
      "calibration_primary_manifest": 0.05810313186763475,
      "calibration_secondary_manifest": 0.4197793699180087,
      "constrained_feasible": false,
      "constrained_objective": Infinity,
      "kl_gt_macro": 0.12859970213316402,
      "kl_gt_uniform_macro": 0.17886820598697734,
      "kl_teacher_macro": 0.062023282598650455,
      "kl_teacher_uniform_macro": 0.04209244276122555,
      "mean_family_kl": 0.09531149236590723,
      "uniform_relative_worst_family_delta": 0.019930839837424906,
      "worst_family_kl": 0.12859970213316402
    },
    "29": {
      "calibration_primary_manifest": 0.0638079974645128,
      "calibration_secondary_manifest": 0.38491790244976676,
      "constrained_feasible": false,
      "constrained_objective": Infinity,
      "kl_gt_macro": 0.1263053653868162,
      "kl_gt_uniform_macro": 0.17886820598697734,
      "kl_teacher_macro": 0.08261309776450032,
      "kl_teacher_uniform_macro": 0.04209244276122555,
      "mean_family_kl": 0.10445923157565826,
      "uniform_relative_worst_family_delta": 0.04052065500327477,
      "worst_family_kl": 0.1263053653868162
    },
    "43": {
      "calibration_primary_manifest": 0.06474217403835307,
      "calibration_secondary_manifest": 0.37629919545724994,
      "constrained_feasible": false,
      "constrained_objective": Infinity,
      "kl_gt_macro": 0.13032047950565545,
      "kl_gt_uniform_macro": 0.17886820598697734,
      "kl_teacher_macro": 0.08252847873146593,
      "kl_teacher_uniform_macro": 0.04209244276122555,
      "mean_family_kl": 0.10642447911856069,
      "uniform_relative_worst_family_delta": 0.04043603597024038,
      "worst_family_kl": 0.13032047950565545
    }
  },
  "uniform_relative_worst_family_delta_would_select": 17,
  "worst_family_kl_would_select": 29
}

## Normal / anomalous (eval d24)

{
  "anomalous": {
    "mean_delta_gt": -0.015515066687354157,
    "mean_delta_teacher": 0.0018724363148810734,
    "mean_kl_gt": 0.04918208579548933,
    "mean_kl_teacher": 0.039621915829202846,
    "n": 4
  },
  "normal": {
    "mean_delta_gt": -0.08469113452717299,
    "mean_delta_teacher": 0.03107479155567594,
    "mean_kl_gt": 0.17052746949051006,
    "mean_kl_teacher": 0.0987091627639578,
    "n": 4
  }
}

## Classification evidence

- Equal-family oracle fails both target-learning gates on calibration (GT macro KL=0.1029, teacher macro KL=0.0664; uniform teacher=0.0421).
- However a non-equal calibration-feasible alpha interval exists [0.31, 0.38] (Case B for reweighted single-head family balance; must not be chosen from evaluation).
- Calibration-feasible alpha interval exists, but evaluation post-hoc feasible alpha interval is empty (generalization / small-sample conflict).
- Teacher fails in all eval categories: ['bottle', 'carpet']
- All seeds {17,29,43} fail teacher macro KL vs uniform on sealed evaluation.
- Mixed causes flagged: ['A', 'C']

## Evaluation contamination boundary

```
current evaluation split status =
used_for_B2_05B_qualification_and_postmortem
```

Any future change to family weights, selection, architecture, loss, targets, or thresholds must follow Protocol 1 (new untouched eval) or Protocol 2 (freeze unqualified; no LSE).

## Artifact

- JSON: `b2_05c0_target_conflict_diagnosis.json` sha256 `8d67ff83977205390ceea2ff0eb71ab3e2863eb2bf149880084399ba86c806b8`
