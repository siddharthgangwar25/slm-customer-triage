# Optional fresh challenge set: deferred

Section 2 asks for at least 100 fresh requests with independent review. No
independent reviewer or reviewed customer dataset is available in this run.
This optional evaluation is therefore **not completed**. The three authored
`examples/demo_requests.json` prompts are demonstrations, not that challenge set.
No challenge accuracy or real-world generalization claim is made.

For a future study, freeze a dated collection protocol before model prediction:

1. Write at least 100 original English requests spanning supported intents,
   unfamiliar topics, ambiguity, typos and instruction-like text. Include
   hospitality requests absent from the taxonomy as `oos`. Use no employer or
   customer messages without their authorized data process.
2. Have a reviewer who did not author the requests assign labels independently,
   without seeing model predictions. Record reviewer identity, date, disagreements
   and adjudication. Ambiguous/multi-intent cases need an explicit expected
   abstention rule rather than forced single-label correctness.
3. Store ID, text, category, proposed label, reviewed label, review status,
   provenance and notes. Label AI-generated material as synthetic. Check exact
   and normalized overlap against public benchmark text, report overlaps, and
   freeze the approved set and hash before generating outputs.
4. Evaluate all frozen candidates under the same protocol. Retain every invalid
   or failed result. Report supported quality, oos review, coverage/error and
   category counts separately; small subgroup estimates need uncertainty.
5. Keep this distinct from CLINC test evidence. A convenience synthetic set is
   not representative customer validation. Tuning after seeing its results
   consumes it; another untouched set is then needed for independent acceptance.

This is a future collection plan, not a request to relabel existing benchmark
errors or relax the original release requirements.
