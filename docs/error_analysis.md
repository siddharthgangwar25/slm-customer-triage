# Error analysis and release decision

The main benefit is supported-intent classification; the main limitation is
open-set behavior. These conclusions use retained validation and the single
completed frozen test, not fresh inference during documentation.

## What training changed

On paired validation, C corrects 2,227 B supported errors and introduces three
regressions. Much of the gain is adherence to the catalog: B invents labels such
as `how_do_you_say_hello`, while C returns `translate`. B has 654 invalid outputs
(653 unknown labels and one invalid JSON); C has one. This is not merely a
syntax improvement. C also recovers many supported requests B labels `oos`.

Remaining confusions include accent versus blocked-account wording, payday
versus income, balance versus credit limit, and accepting reservations versus
making them. The three supported regressions include a restaurant reservation
boundary, a fraud request misread as a damaged card, and fuel needed for a trip
misread as distance. Official labels remain unchanged even where wording is
ambiguous. The [40-example validation review](../reports/three-way-qwen3-06b-v1/error_analysis.md)
and its linked JSON explain the deterministic sample and each observation;
it is assistant-reviewed analysis, not independent reannotation.

## Raw oos behavior versus policy review

C's raw oos correctness declines from B's 67/100 to 55/100 on validation and
from 718/1,000 to 470/1,000 on test. It overextends familiar intents to unsupported
requests, such as projections, inventory or adjacent maintenance tasks. Only
100 training oos examples are available. The separate baseline gate rejects
many of these, but cannot guarantee open-set correctness.

On final test, C reviews 896/1,000 oos requests and routes 104 incorrectly.
Of 185 total routed errors, 81 are supported-label mistakes and 104 are oos
misroutes. There are 44 invalid outputs, 42 on oos and two on supported requests.
Invalid output can cause review under the policy, but that is not a correct raw
classification. C's oos recall misses the fixed 900/1,000 requirement by four;
uncertainty intervals do not convert this failed point-estimate rule into a pass.

Compared with validation, test has a substantially larger oos fraction. Coverage
is therefore workload-dependent, not a constant property of the model. A fails
the test routing-error constraint; B reviews everything. The honest conclusion
is improved supported classification with no qualified automatic release.
See the [final report](../reports/milestone5-final-v1/README.md).

## Backend and operational failures

The alternative vLLM runtime changes five C validation labels, three after the
gate. Two routed errors improve; one gated correct label regresses; one gated
mistake and one oos misroute change labels. No request changes between route and
review, illustrating why equal aggregate coverage is insufficient for parity.
[All changed examples](../reports/vllm-validation-v1/C-output-changes.json).
The variant is not adopted or substituted after test exposure.

At concurrency 4/8, the verified Transformers service mostly rejects requests as
busy. Reporting their fast rejection latencies as successful response speed
would hide overload. No throughput scaling or real customer robustness is claimed.

## Future work, without reopening this test

A future experiment should obtain broader independently reviewed unsupported
traffic, define ambiguity/multi-intent handling and re-evaluate the shared-gate
assumption. Better calibration or alternative gates need their own validation.
Improve serving concurrency only after an evaluated candidate and capacity
target exist. Preserve original evidence and disclose test exposure; do not
search the current test set for a threshold/backend that passes.
