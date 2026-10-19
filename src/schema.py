"""Closed review question shared by the teacher prompt and the student."""

QUESTIONS = {
    "review": {
        "type": "choice",
        "instructions": "What should a reviewer do with this change?",
        "criteria": {
            "approve": "Safe to merge. The change is tested, or it does not change behavior (rename, docs, format, unused import).",
            "request_changes": "Behavior is wrong, incomplete, or a behavior change has no test.",
            "escalate": "A secret or credential is added or logged, authentication or authorization is bypassed, data is destroyed, or TLS verification is turned off.",
        },
    }
}

LABELS = ("approve", "request_changes", "escalate")

RUBRIC = """
escalate: a secret or credential is added or logged, authentication or authorization is bypassed, data is destroyed, or TLS verification is turned off.
request_changes: behavior is wrong, incomplete, or a behavior change has no test.
approve: safe to merge. The change is tested, or it does not change behavior (rename, docs, format, unused import).
""".strip()


def render_state(state: str) -> str:
    criteria = "\n".join(
        f"- {key}: {text}" for key, text in QUESTIONS["review"]["criteria"].items()
    )
    return (
        f"{QUESTIONS['review']['instructions']}\n\n"
        f"Options:\n{criteria}\n\n"
        f"Change:\n{state.strip()}"
    )
