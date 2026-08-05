# B2-08 Paper Results Summary

Status: paper results and evidence index closed locally.

This document summarizes the frozen B2 DLCM + LSE experiment for paper writing.
It does not introduce new training, evaluation, model checkpoints, or scientific
outputs. All values below are copied from tracked B2 evidence documents.

## Method Summary

The B2 line extends the VisualAD fixed equal-depth fusion over ViT-Large layers
`[6, 12, 18, 24]` with a Dynamic Layer Contribution Module (DLCM). DLCM predicts
input-dependent layer fusion weights at prediction depths `[12, 18, 24]` from
18-D layer descriptors, depth/layer embeddings, and a shared set-context MLP.

The accepted DLCM is the V5 uniform-anchored deployment. It keeps the frozen C3
dynamic deployment checkpoint and applies a post-hoc global calibration:

```text
w_deploy(x; beta) = (1 - beta) * uniform + beta * w_dynamic(x)
```

The frozen scalar is:

```text
beta* = 0.54
```

The downstream LSE stage is trained only after the accepted V5 gate is wired and
all prerequisites are materialized. LSE uses the accepted DLCM V5 artifact chain
instead of a manually supplied checkpoint path.

## Primary Frozen Identities

| Item | Value |
|---|---|
| V5 deployment identity | `c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd` |
| Accepted DLCM identity | `0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116` |
| Final decision identity | `6fb60a82d01f987930070aeee75639524512ad481064369b2f06ac99f96ae0a8` |
| Final evidence identity | `bbc3708a8ddcd3b2965ec9e758af1a7bf30a360cdbbc5ff86be911cfbe872e02` |
| Accepted V5 checkpoint SHA256 | `12b9192643d457eb07745391b68cfa5afe48ec6165b28091bdabde29ec3ece4f` |
| LSE qualification identity | `0f08407ade40fb8e447649c80606fbf7d7c39f3030b99307a445f3df27688b14` |
| Accepted LSE identity | `3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa` |
| Accepted LSE checkpoint SHA256 | `e6e5a4dbd7471ef9e52430eab9533f8edda57ca76ead2ffbed034044805b1c98` |
| B2 phase final closure identity | `2b1e74c13bba260a9f62c4167b322ae067ecce34fc86a92ae66e1a71b0f3073d` |

## Result Table

| Stage | Scope | Result | Paper-use note |
|---|---|---:|---|
| DLCM V1 | Original dual GT/teacher target | unqualified | Localized but teacher target fidelity failed; retained as negative evidence. |
| DLCM V2/V3/V4 | Decoupled, category-robust, uniform-relative repairs | unqualified | Development failed on carpet GT category gate. |
| DLCM V5 | Uniform-anchored beta calibration | accepted | `beta*=0.54`; accepted identity frozen after Final qualification. |
| LSE training | First controlled run | completed | 30 epochs, best epoch 22, seed 111. |
| LSE qualification | Calibration NLL gate | qualified | `0.4768362585455179 <= 0.5` over 16 evaluated rows. |
| LSE accepted artifact | Accepted closure | frozen | Accepted LSE identity `3daf...01fa`. |
| B2 final closure | Local evidence handoff | frozen | Phase final closure identity `2b1e...073d`. |

## LSE Qualification Metrics

| Metric | Value |
|---|---:|
| Calibration NLL | `0.4768362585455179` |
| Qualification threshold | `0.5` |
| Evaluated rows | `16` |
| Required depths | `12, 18` |
| Depth 12 NLL | `0.7660192847251892` |
| Depth 12 MAE | `0.4048725925385952` |
| Depth 12 RMSE | `0.5041921386533236` |
| Depth 12 Brier | `0.03221412755399544` |
| Depth 12 ECE | `0.17947600036859512` |
| Depth 18 NLL | `0.18765321373939514` |
| Depth 18 MAE | `0.20543629676103592` |
| Depth 18 RMSE | `0.2283385445394299` |
| Depth 18 Brier | `0.024174126336846347` |
| Depth 18 ECE | `0.15546763315796852` |

## Suggested Paper Wording

We evaluate a dynamic layer contribution module that predicts per-input fusion
weights over the same ViT-Large layers used by the fixed VisualAD baseline. The
accepted variant uses a uniform-anchored calibration of the frozen dynamic
weights, with a single global scalar selected on Calibration and then frozen for
Development/Final. This preserves the equal-weight baseline as the beta-zero
endpoint while allowing input-dependent weighting when supported by the
calibration gate.

For the LSE extension, the training entrypoint is gated by the accepted DLCM V5
artifact rather than a manual checkpoint path. The first controlled LSE run
qualified under the frozen calibration NLL gate, with calibration NLL
`0.4768362585455179` against a threshold of `0.5`, and the accepted LSE artifact
was frozen under identity
`3dafdde6309599d7e82ca6da07db4efbdb09f16105262351c890c514277f01fa`.

## Boundary Statement

```text
B2-08 training_started = false
B2-08 evaluation_started = false
B2-08 final_content_accessed = false
B2-08 model_artifact_generated = false
tracked .pt = 0
pushed = false
PR = false
```
