# B2-06A LSE Accepted-Gate Wiring / Preflight Closure

## Status

B2-06A only wires the LSE entrypoint to the accepted V5 deployment artifact. It does not start LSE training and does not generate an LSE checkpoint.

## Frozen identities

- beta*: `0.54`
- V5 deployment identity: `c56248c9ff6021fc16cf4792d87afeebf1bb8f6d45859f7c26017830dcf0e0bd`
- accepted V5 identity: `0c1a411317f212e5deb29040d184d57aead8a6f862fe3146937db99d1f365116`
- Final decision identity: `6fb60a82d01f987930070aeee75639524512ad481064369b2f06ac99f96ae0a8`
- Final evidence identity: `bbc3708a8ddcd3b2965ec9e758af1a7bf30a360cdbbc5ff86be911cfbe872e02`
- Calibration A/B identity: `cae406c91ec392ffd7cc6d48ec2f0c94ab78d78f905cbfe904287842a7a7278a`

## Gate behavior

The LSE entrypoint now requires an accepted V5 manifest before dry-run, preflight, or training can proceed. The gate verifies accepted identity, V5 deployment identity, Final decision identity, Final evidence identity, beta*, and calibration identity. A DLCM checkpoint path must be under the accepted artifact reference root; a manual C4B checkpoint path fails closed.

## Preflight result

`configs/rad/lse_b2_accepted_v5.yaml` resolves the accepted Final manifests and passes the accepted gate. Readiness remains false because formal LSE prerequisites are still absent:

- `dlcm_checkpoint`
- `train_gain_targets`
- `calibration_gain_targets`
- `train_cache`
- `calibration_cache`
- `descriptor_stats`

This is the intended stop point for B2-06A.

## Verification

- LSE unit/focused tests: `12 passed`
- Legacy `configs/rad/lse.yaml` dry-run: exit `2`, `B2_LSE_ACCEPTED_MANIFEST_REQUIRED`
- B2 accepted preflight: exit `2`, accepted gate passed, readiness false
- Ruff: clean
- scoped mypy: clean
- running LSE process count: `0`
- tracked `.pt`: `0`
- LSE checkpoint directory: absent
