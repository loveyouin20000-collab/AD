# B3-04 Exit Target Redefinition / Positive-Signal Contract Plan

Goal: define a legal conservative source for positive early-exit labels without
reading Final content or training an exit policy.

Contract:

- Inputs are B3-02 calibration trace, B3-02 latency proxy, and accepted B2/B3
  identities.
- A candidate positive exit label may only be assigned when:
  - predicted remaining gain is below a fixed threshold;
  - predicted sufficiency probability is above a fixed threshold;
  - latency proxy has positive savings versus full depth.
- Thresholds are frozen by contract before any training.
- If no candidate positives exist, training remains locked.

Default conservative thresholds:

- `max_predicted_remaining_gain = 0.10`
- `min_predicted_sufficiency_probability = 0.50`

Boundary:

- no training
- no evaluation
- no checkpoint generation
- no Final content access
- tracked `.pt` remains zero
