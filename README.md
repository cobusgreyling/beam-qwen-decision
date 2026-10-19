<p align="center">
  <img src="assets/header.jpg" alt="A small brass scale under a single shaft of light, one pan holding a glowing chip" width="100%" />
</p>

# Beam labels. A 0.8B model decides.

**The training run has not been started. This repository is the plan.**

**A 501B teacher writes closed labels. Qwen3.5-0.8B learns to score those options and return a probability. The small model never has to write the reply.**

**Tags:** `Beam` · `Qwen3.5-0.8B` · `decision model` · `LoRA` · `Unsloth` · `OpenCode`

---

A router, a guardrail, or a ticket desk does the same job all day. Read a state. Answer a question whose options you already wrote down. Hand the label back to code.

That job does not need a 501B model on the hot path. It needs a small model that has seen *your* decisions. [Reflection Beam](https://huggingface.co/reflection/Beam-1.0-501B-A23B) is the teacher. [Qwen3.5-0.8B](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth) is the student. Beam reads the hard cases once. The 0.8B model answers them after that.

This post is the plan for that lab. The accuracy table further down is Unsloth’s published run on public sets. It is not a number from this repository.

---

## What the small model is for

A decision model fills a slot you declared before the call.

| Type | What it returns | Example |
|------|-----------------|---------|
| **Choice** | One key from a set you defined, plus a probability for each key | `approve` / `request_changes` / `escalate` |
| **Noul** | Yes or no, as a probability | “Does this diff touch auth?” |
| **Score** | A level on your rubric | `not urgent` / `soon` / `today` |

The reply to the customer, the patch, the commit message: those stay with a generative model. The 0.8B model only judges. Code owns the threshold. Low probability means ask a person, or ask Beam again.

Jev, Laya, and Cloudflare Clef are the same shape. This lab trains that shape on your own rows, on a checkpoint you can keep.

---

## Why 0.8B, and why a head

Qwen3.5-0.8B is large enough to read a diff or a ticket, and small enough to train on a free T4. Unsloth’s decision-model recipe puts a small scoring head on the language model, the same design as Clef. The model reads the state and the options once. The head scores every option. It does not spend the decode on writing the word `escalate`.

That matters at this size. On closed questions, a classification head beats “please answer with one label” for models under about 2B. The probability comes from the scores, then you calibrate it. You do not ask the model to invent a confidence number in prose.

Unsloth’s October 2026 runs, one epoch of LoRA on a public mix, then a 3,000-row holdout:

| Model | typed-decisions | BANKING77 | CLINC150 | Holdout |
|-------|----------------:|----------:|---------:|--------:|
| Qwen3.5-0.8B | 36% → **73%** | 7% → **74%** | 19% → **76%** | **78%** |
| Qwen3.5-2B | 33% → **78%** | 1% → 58% | 1% → 62% | **81%** |

The 0.8B run used about 4 GB and 42 minutes. The 2B run used about 8 GB and landed at 81% holdout. Start at 0.8B. Move to 2B only if the 0.8B model misses the human slice described below.

Settings they say match that table: 2 epochs, LoRA rank 16, learning rate 2e-4, 4-bit. Checkpoint id in their loader: `unsloth/Qwen3.5-0.8B`. Guide: [Train your own Decision Model](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth).

Those sets are banking intents and generic typed questions. They show the size works. They do not predict accuracy on your pull requests.

---

## How Beam is reached

Beam’s own platform asks for a card before it will issue an API key. This lab does not use that key.

OpenCode serves **Beam Free** for a limited time, model id `opencode/beam-free`, at no token charge. Install the app and run it inside the repository:

```bash
opencode run --model opencode/beam-free "Label this diff. Return JSON only."
```

A direct HTTP call to the same endpoint, without the OpenCode app, returns 403: the free tier can only be used from within OpenCode. The labeling script is an OpenCode session, or a batch of prompts you paste there. It is not a Python client with a Reflection key.

Two limits from OpenCode’s model notes. Prompts are kept for 30 days for safety, and they are not used for training. Strip secrets, tokens, and customer data before a state goes into a prompt. The free window has no published end date.

The Apache 2.0 weights are on Hugging Face as `reflection/Beam-1.0-501B-A23B`, plus FP8 and NVFP4 checkpoints. NVFP4 needs on the order of 350 GB of GPU memory. That is a serving cluster. It is the wrong machine for writing a few thousand labels.

---

## The rows, in the order they are worth

Beam is a labeler for gaps. It is not the author of the dataset.

1. **Outcomes you already have.** A merged or rejected pull request, a test that failed, a ticket a person escalated. These rows are gold. Beam does not relabel them.
2. **Beam, twice, on real states with no outcome.** One closed question per row. Keep the row only when both passes return the same label. Disagreements go to a person, or they are dropped.
3. **A human slice before the big run.** Label 100 to 200 real states yourself. Score Beam against that slice. Generate the rest only if Beam is right often enough that you would trust the student to copy it.
4. **Synthetic states only for a rare class.** If `escalate` almost never appears, ask Beam to write states that belong there, then label them with the same question. Mark those rows `synthetic`. They stay the minority.

Balance the classes. A log that is 95% `approve` teaches a keyword matcher. A few thousand clean rows are enough for one narrow decision. Split by time, or by repository, so the model cannot memorize one codebase’s wording.

Each training row is the shape Unsloth’s decision trainer expects:

```json
{
  "state": "PR changes auth middleware and adds no test.",
  "questions": {
    "review": {
      "type": "choice",
      "instructions": "What should happen to this diff?",
      "criteria": {
        "approve": "Safe to merge as-is",
        "request_changes": "Needs a test or a smaller diff",
        "escalate": "Security or data-loss risk"
      }
    }
  },
  "gold": { "review": "request_changes" },
  "source": "beam-agreement"
}
```

`source` is for you. The trainer uses `state`, `questions`, and `gold`. Choice gold is a key in `criteria`. A noul gold is `true` or `false`. A score gold is the level index, starting at 0.

Beam’s reasoning channel stays out of the target. So does any confidence number Beam writes in prose. After training, fit the 0.8B probabilities on the human slice. That slice is also the frozen test. The trainer never updates on it.

---

## The training pass, when it is run

The notebook to start from is Unsloth’s [Qwen3.5 decision notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)-Decision.ipynb). Change the model name to `unsloth/Qwen3.5-0.8B` for the first run. A free Colab T4 covers the 4 GB budget.

The loop, once the JSONL exists:

1. Score the untouched 0.8B model on the human slice. That is the before number.
2. LoRA-train the decision head on the gold rows plus the Beam rows that survived the agreement filter.
3. Calibrate on the human slice.
4. Score the same slice again. Publish that pair, and no other accuracy claim.

A new ticket is one call. The model returns the option key and a probability for every option. Code applies the threshold you wrote down.

---

## What this draft is not

There is no adapter in this folder. There is no measured before/after on a real repository. The 78% figure belongs to Unsloth’s public mix, and this post cites it as theirs.

The next concrete piece of work is the question schema and the 100-row human slice. Beam labeling comes after that slice says the teacher is worth copying.
