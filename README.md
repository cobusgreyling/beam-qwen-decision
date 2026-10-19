<p align="center">
  <img src="assets/header.jpg" alt="A small brass scale under a single shaft of light, one pan holding a glowing chip" width="100%" />
</p>

# Beam labels. A 0.8B model decides.

**Beam Free labelled 34 short diffs. I trained Qwen3.5-0.8B, with a rank-16 LoRA and a 3-way head, on this Mac. On 18 frozen diffs the score moved from 7 / 18 to 13 / 18.**

**Tags:** `Beam` · `Qwen3.5-0.8B` · `decision model` · `LoRA` · `Unsloth` · `OpenCode`

---

Beam is Reflection's first open-weight model. [Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B) has 501 billion parameters in total and about 23 billion active per token. The model card lists 501.3 billion and 23.4 billion. It is built for coding, reasoning, and agentic work, with a 1 million token context, and the weights are Apache 2.0.

Those parameters sit in many smaller sub-networks, called experts. A router picks a few of them for each token. Beam has 256 experts and runs 8 of them. The router follows the DeepSeek-V3 group limit: it scores all 256, splits them into 8 groups of 32, keeps the 4 best groups, and then takes the top 8 experts inside those groups. There is no shared expert. The first 3 of the 52 layers are dense. The other 49 are mixture-of-experts.

About 4.7% of the weights do any work on a given token. The [Hugging Face post](https://huggingface.co/blog/reflection-beam) says it directly: the model computes each token with about 23 billion parameters, and it runs at the speed of a 23B model. That is a compute claim. I would not read it as the same invoice as serving a dense 23B model.

The catch is memory. All 256 experts still have to be loaded, because the next token can be routed somewhere else. Deployment needs the memory of a 501B model. The same post says the NVFP4 checkpoint shrinks the BF16 weights from about 1 TB to about 280 GB, and that two B200s (360 GB) can hold those weights with little room left for activations and the KV cache. They suggest planning for four B200s (720 GB). The [vLLM recipe](https://recipes.vllm.ai/reflection/Beam-1.0-501B-A23B) lists the NVFP4 weights at 294 GB on disk and a minimum of 353 GB of GPU memory, which includes headroom. FP8 is 503 GB on disk and 604 GB minimum. BF16 is 1,003 GB on disk and 1,204 GB minimum.

The Hugging Face title is the brief I wanted: [Welcome Beam: a 501B open model you can own, customize, and scale](https://huggingface.co/blog/reflection-beam).

## The goal

I wanted a prototype that made that concrete.

The job I care about is small. Read a state. Answer a question whose options were written down in advance. Hand the label back to code.

That job does not need a 501B model on the usage path. It needs a small model that has seen my decisions. Beam reads the hard cases once and writes closed labels. The small model answers them from then on.

Replies, patches, and commit messages stay with a generative model. The small model only judges. My code owns the threshold. When the probability is low, a person looks at it, or I ask Beam again.

## What a decision model returns

The small model fills a slot I declared before the call. Three shapes:

| Type | What comes back | Example |
| --- | --- | --- |
| Choice | One option from a set I define, with a probability for each | `approve`, `request_changes`, or `escalate` on a pull request |
| Yes/no | A single probability | Does this diff touch auth? |
| Score | A level on my rubric | `not urgent`, `soon`, or `today` |

Unsloth's decision recipe, and the Clef head it follows, scores the options. The model reads the state and the options once. The head scores every option. It never spends tokens writing out the word `escalate`. For models under about 2B, that works better than asking for the label in prose.

This prototype trained one Choice question, `review`, with the three options above. Yes/no and Score are the rest of the interface. They were not in this run.

The probabilities in `outputs/results.json` are a softmax over the three scores. I have not calibrated them. Calibration on a slice the trainer never sees is the next step, once that slice is bigger than 18 rows.

## Where Beam lives, and how I reached it

There are three ways to access Beam.

The first is Reflection's own platform, an OpenAI-compatible API at `https://api.reflection.ai`. Getting a key there means adding a credit card. I skipped it.

The second is OpenCode, a coding agent that runs in the terminal. For a limited time it offers Beam at no token charge under the name `opencode/beam-free`. The free window has no published end date. This is the one I used. I open OpenCode against the repository and send Beam a request like this:

```bash
opencode run --model opencode/beam-free "Label this diff. Return JSON only."
```

The free model only works inside OpenCode. A direct call to the same HTTP endpoint, without the app, comes back 403 with `FreeTierError`: the free tier can only be used from within OpenCode. All the labelling in this repo goes through OpenCode sessions. `src/label_beam.py` shells out to the CLI, batches six states, and keeps the JSON array from the reply.

OpenCode keeps prompts for 30 days for safety checks. It does not train on them. I still remove passwords, keys, and customer data before I send anything. The states in this repo are short summaries I wrote myself, so there was nothing private in the prompts.

The third way is to download the weights and run them. They are on Hugging Face under Apache 2.0. Even the NVFP4 checkpoint needs on the order of 350 GB of GPU memory before there is room for a KV cache. That is a cluster of high-end GPUs. It is far too much just to label a few thousand examples, which is all I needed Beam for.

## The Qwen model

The student is [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), Apache 2.0. It is big enough to read a short diff and small enough to train on a laptop.

Unsloth published the decision-model recipe I started from: [Train your own Decision Model](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth). Their loader id is `unsloth/Qwen3.5-0.8B`. To match the table on that page they say 2 epochs, LoRA rank 16, learning rate 2e-4, and 4-bit, with the new head at 1e-4. The same page also describes an earlier run at rank 64 for one epoch. On their public mix the 0.8B model reached a 78% holdout, in about 4 GB, in about 42 minutes. Those numbers are Unsloth's, on banking intents and generic typed questions. They are not this repository.

This run loaded `Qwen/Qwen3.5-0.8B`. The trainer is `src/train.py`, not the Unsloth notebook, because that notebook expects a CUDA GPU and this machine is an Apple M2 Pro with 16 GB of unified memory. fp16 on MPS produced NaN hidden states in Qwen3.5's hybrid stack, so the run is float32. There was no 4-bit load and no Colab T4. The vision tower stayed on CPU. A rank-16 LoRA sits on the text attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`). A new linear head reads the last non-pad hidden state and scores the three labels.

With 34 training rows I used 4 epochs. Unsloth suggests 3 or 4 when the set is small. LoRA learning rate 2e-4, head learning rate 1e-4, batch size 2, max length 384, seed 3407. Train loss went from 1.47 to 0.19. Once the weights were on disk, the script finished in about 80 seconds.

## What I want the training rows to be

I rank rows by how much I trust them.

1. Outcomes I already have, such as a merged or rejected pull request, a test that failed, or a ticket a person escalated. Those rows are gold. Beam does not relabel them.
2. Beam labels on real states that have no outcome yet. I run Beam twice and keep a row only when both runs agree. Disagreements go to a person, or I drop them.
3. A human slice of 100 to 200 states I label myself. That slice is the test. The trainer never sees it. I also use it to see whether Beam is right often enough that I would trust a student to copy it.
4. Synthetic states for rare classes only, tagged `synthetic`. If `escalate` almost never appears, I can ask Beam to write states that belong there, then label them with the same question. Those rows stay the minority.

I do not train on Beam's reasoning text, and I do not train on a confidence number Beam writes in prose. The target is the closed label. After training, I want the small model's probabilities calibrated on the human slice. Split by time, or by repository, so the model cannot memorise one codebase's wording.

## What this prototype actually used

This lab is smaller than that list, and it is earlier in the pipeline.

Every state here is a short diff summary I wrote by hand. There are 36 of them in `data/train_states.jsonl` (`t01`–`t36`) and 18 frozen ones in `data/gold_test.jsonl` (`g01`–`g18`). None of them is a live pull request, a merge, or a rejection from a repository. There is no outcome file Beam was told to leave alone. The whole set is written, including the common classes, so it is not a rare-class supplement tagged `synthetic`.

What I did run is the agreement check, and a small human slice.

Beam Free labelled the 36 training states twice. I kept a row only when both passes agreed: 34 rows in `data/beam_train.jsonl`, source `beam-agreement`. Two rows went to `data/beam_dropped.jsonl`:

- `t08`. CSRF middleware removed, and `Access-Control-Allow-Origin` set to `*` on the account API. One pass said `request_changes`, the other `escalate`.
- `t25`. `userName` renamed to `user_name`, and the existing tests pass. One pass said `approve`, the other `request_changes`.

The kept rows split as 9 `escalate`, 11 `approve`, and 14 `request_changes`.

The 18 gold diffs are the human slice. Six of each class, worded differently from the training states. I labelled them before training. The trainer never saw them. Eighteen is a lab, not the 100 to 200 I want before I would trust the score, and it is not a benchmark.

One kept row, trimmed to the fields the trainer reads. The exact criteria are in `data/RUBRIC.md`.

```json
{
  "state": "The import endpoint does pickle.loads on the raw request body.",
  "questions": {
    "review": {
      "type": "choice",
      "instructions": "What should a reviewer do with this change?",
      "criteria": {
        "approve": "Safe to merge. The change is tested, or it does not change behavior (rename, docs, format, unused import).",
        "request_changes": "Behavior is wrong, incomplete, or a behavior change has no test.",
        "escalate": "A secret or credential is added or logged, authentication or authorization is bypassed, data is destroyed, or TLS verification is turned off."
      }
    }
  },
  "gold": { "review": "request_changes" },
  "source": "beam-agreement"
}
```

That row is `t07`. Both Beam passes said `request_changes`. The written `escalate` rule names a secret, an auth bypass, destroyed data, or TLS verification turned off. `pickle.loads` on a request body is dangerous, and it is not one of those four, so the agreed label follows the rubric as written.

## The fine-tune

I scored the untouched model on the 18 gold diffs first. That baseline is the mean log-probability of each label as a continuation of the rendered state plus `Label:`. It is the base model's preferred label token, before any LoRA or head. Then I LoRA-trained on the 34 agreement rows only. The gold file was not in the loss. Then I scored the same 18 diffs again.

```bash
python src/label_beam.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train.py
```

`train.py` writes `outputs/results.json` and `outputs/adapter`.

| Check | Base Qwen3.5-0.8B | LoRA + decision head |
| --- | ---: | ---: |
| Gold test | 7 / 18 | **13 / 18** |
| `approve` | 0 / 6 | **6 / 6** |
| `request_changes` | 6 / 6 | **6 / 6** |
| `escalate` | 1 / 6 | 1 / 6 |

The base model answered `request_changes` on 17 of the 18 diffs. Its one other answer was `escalate`, on `g06`. Training moved `approve` from 0 / 6 to 6 / 6. `request_changes` stayed perfect. `escalate` stayed at 1 / 6, and it was a different row: the base model had `g06`, and the trained head has `g01`.

`g01` is a `.env` file that contains a database password. That matches the rubric's secret rule, and the head got it. The other five gold `escalate` rows were called `request_changes`:

- `g02`, `require_admin` becomes `if user.role == "admin" or True`
- `g03`, `POST /cleanup` runs `shutil.rmtree` on the customer uploads directory
- `g04`, the bank callback uses `requests.get(url, verify=False)`
- `g05`, a trace line prints the `Authorization` header
- `g06`, the calculator route passes `request.form['expr']` to `eval`

Beam, on one pass over the same 18 diffs, matched my gold label on **15 / 18**. Those Beam labels were not training targets. The three misses:

- `g06`, gold `escalate`, Beam `request_changes`. The student made the same call. The written escalate rule does not name code execution.
- `g11`, the webhook returns HTTP 200 before the signature check. Gold `request_changes`, Beam `escalate`.
- `g18`, an unreachable branch is deleted and the suite still passes. Gold `approve`, Beam `request_changes`.

On `g02`, `g03`, `g04`, and `g05`, Beam matched the gold `escalate` label and the student did not. Nine escalate rows in a 34-row file were not enough for that class. The gain in this lab is the `approve` class.

Full rows: [`outputs/results.json`](outputs/results.json). Beam's pass over the gold file: [`outputs/beam_on_gold.json`](outputs/beam_on_gold.json). Adapter and head: [`outputs/adapter`](outputs/adapter).

The 78% figure above is still Unsloth's holdout on a public mix. The number from this repository is 7 / 18 to 13 / 18 on these diffs.
