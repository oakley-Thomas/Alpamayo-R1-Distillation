from __future__ import annotations


def token_trajectory_alignment_score(cot_text: list[str], trajectory) -> float:
    if not cot_text:
        return 0.0
    avg_len = sum(len(text.split()) for text in cot_text) / len(cot_text)
    return float(avg_len / max(trajectory.shape[1], 1))
