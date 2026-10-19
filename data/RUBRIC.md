# Review rubric

One closed question, `review`.

| Label | When it is correct |
| --- | --- |
| `escalate` | A secret or credential is added or logged, authentication or authorization is bypassed, data is destroyed, or TLS verification is turned off. |
| `request_changes` | Behavior is wrong, incomplete, or a behavior change has no test. |
| `approve` | Safe to merge. The change is tested, or it does not change behavior (rename, docs, format, unused import). |

`data/train_states.jsonl` has no labels. Beam Free labels that file twice, and a row is kept only when both passes agree.

`data/gold_test.jsonl` is the frozen human slice. It is scored before and after training. It is not a training source.
