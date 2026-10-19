# This run

Frozen gold test, 18 diffs. The trainer did not see them.

| Check | Base Qwen3.5-0.8B | LoRA + decision head |
| --- | ---: | ---: |
| Gold test | 7 / 18 | 13 / 18 |
| approve | 0 / 6 | 6 / 6 |
| request_changes | 6 / 6 | 6 / 6 |
| escalate | 1 / 6 | 1 / 6 |

Beam Free, one pass on the same 18 diffs, matched the gold label on 15 / 18.

Training rows are the 34 states where two Beam passes agreed. Two states were dropped. Counts in the training file: escalate 9, approve 11, request_changes 14.

Device: Apple M2 Pro, float32. fp16 on MPS returned NaN hidden states. Rank-16 LoRA, 4 epochs, train loss 1.47 to 0.19.
