"""Fine-tune Qwen3.5-0.8B as a review decision model.

The language-model stack gets a rank-16 LoRA. A new linear head scores the
three review options from the last hidden state. The head does not generate text.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schema import LABELS, render_state  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-0.8B"
SEED = 3407
MAX_LENGTH = 384
EPOCHS = 4
LR = 2e-4
HEAD_LR = 1e-4


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DecisionModel(nn.Module):
    def __init__(self, language_model, hidden_size: int, n_labels: int):
        super().__init__()
        self.language_model = language_model
        self.head = nn.Linear(hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state
        lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        return self.head(pooled.float())


def encode(tokenizer, states: list[str], dev: torch.device):
    text = [render_state(state) for state in states]
    batch = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return batch["input_ids"].to(dev), batch["attention_mask"].to(dev)


@torch.no_grad()
def predict(model, tokenizer, rows: list[dict], dev: torch.device) -> list[dict]:
    model.eval()
    results = []
    for row in rows:
        input_ids, attention_mask = encode(tokenizer, [row["state"]], dev)
        logits = model(input_ids, attention_mask)[0].float()
        probabilities = torch.softmax(logits, dim=-1)
        index = int(probabilities.argmax().item())
        results.append(
            {
                "id": row["id"],
                "gold": row["gold"]["review"],
                "answer": LABELS[index],
                "probabilities": {
                    label: round(float(probabilities[i]), 4) for i, label in enumerate(LABELS)
                },
            }
        )
    return results


def accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    hits = sum(row["answer"] == row["gold"] for row in rows)
    return hits / len(rows)


@torch.no_grad()
def zero_shot(language_model, lm_head, tokenizer, rows: list[dict], dev: torch.device) -> list[dict]:
    """Score the base model by the label token it prefers, before any training."""
    language_model.eval()
    results = []
    label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    for row in rows:
        prompt = render_state(row["state"]) + "\n\nLabel:"
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        input_ids = encoded["input_ids"].to(dev)
        scores = []
        for ids in label_ids:
            continuation = torch.tensor([ids], device=dev)
            sequence = torch.cat([input_ids, continuation], dim=1)
            hidden = language_model(input_ids=sequence, use_cache=False).last_hidden_state
            logits = lm_head(hidden)[0, input_ids.size(1) - 1 : -1].float()
            log_probs = F.log_softmax(logits, dim=-1)
            target = continuation[0]
            scores.append(float(log_probs.gather(1, target.unsqueeze(1)).sum() / max(len(ids), 1)))
        index = max(range(len(scores)), key=lambda i: scores[i])
        results.append(
            {
                "id": row["id"],
                "gold": row["gold"]["review"],
                "answer": LABELS[index],
            }
        )
    return results


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    dev = device()
    train_rows = read_jsonl(ROOT / "data" / "beam_train.jsonl")
    test_rows = read_jsonl(ROOT / "data" / "gold_test.jsonl")
    if not train_rows:
        raise SystemExit("data/beam_train.jsonl is empty. Run src/label_beam.py first.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # fp16 on MPS produces NaN hidden states in this hybrid stack.
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    base = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype)
    base.model.visual.requires_grad_(False)
    base.model.visual.to("cpu")
    language_model = base.model.language_model.to(dev)
    lm_head = base.lm_head.to(dev)
    before = zero_shot(language_model, lm_head, tokenizer, test_rows, dev)
    print(f"before {accuracy(before):.3f}", flush=True)

    language_model = get_peft_model(
        language_model,
        LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    hidden = base.config.text_config.hidden_size
    model = DecisionModel(language_model, hidden, len(LABELS)).to(dev)
    model.head.to(dtype=torch.float32)

    head_params = list(model.head.parameters())
    lora_params = [p for p in model.language_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=0.01,
    )

    order = list(range(len(train_rows)))
    model.train()
    for epoch in range(EPOCHS):
        random.shuffle(order)
        total = 0.0
        seen = 0
        for start in range(0, len(order), 2):
            batch_index = order[start : start + 2]
            batch = [train_rows[i] for i in batch_index]
            input_ids, attention_mask = encode(tokenizer, [row["state"] for row in batch], dev)
            labels = torch.tensor(
                [LABELS.index(row["gold"]["review"]) for row in batch],
                device=dev,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask).float()
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
        print(f"epoch {epoch + 1} loss {total / seen:.4f}", flush=True)

    after = predict(model, tokenizer, test_rows, dev)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    report = {
        "model": MODEL_ID,
        "device": dev.type,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "epochs": EPOCHS,
        "lora_r": 16,
        "learning_rate": LR,
        "head_learning_rate": HEAD_LR,
        "before_accuracy": round(accuracy(before), 4),
        "after_accuracy": round(accuracy(after), 4),
        "before": before,
        "after": after,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    adapter = out / "adapter"
    model.language_model.save_pretrained(adapter)
    torch.save(model.head.state_dict(), adapter / "decision_head.pt")
    tokenizer.save_pretrained(adapter)
    print(json.dumps({k: report[k] for k in ("before_accuracy", "after_accuracy", "train_rows")}, indent=2))


if __name__ == "__main__":
    main()
