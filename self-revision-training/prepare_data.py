"""
Build the two SFT stages described in Section 2.1 of the paper from
D_revision = { (x, y_init, P_r, y_revised) } collected by
`self_critique_pipeline.py`.

Stage 1 (generation loss):
    prompt     = x
    completion = y_init + "\n\n" + P_r + "\n\n" + y_revised
    Uses ONLY incorrect-init tuples, so the model learns the full
    mistake -> revision trajectory as one stream. Stage 1 takes *all*
    incorrect-init tuples left after Stage 2 reserves its small quota — its
    size is not fixed, it scales with however much data was collected.

Stage 2 (revision loss, ~1k by default):
    prompt     = x + "\n\n" + y_init + "\n\n" + P_r
    completion = y_revised
    A small fixed-size set (--stage2_size). Correct-init tuples — collected
    only when the pipeline ran with --revise_correct — are placed here first
    since Stage 1 cannot use them; the quota is then filled with incorrect-init.

Response fields are cleaned before use (degenerate repeating tails trimmed,
sycophantic chat openings stripped) — disable with --no_clean.

Outputs two JSON files (lists of {"prompt", "completion"} records)
ready to be consumed by `sft/sft.py`.
"""

import argparse
import json
import os
import random
import re
from typing import Any, Dict, List, Tuple


# --------------------------------------------------------------------------
# Response cleaning
# --------------------------------------------------------------------------

# Sycophantic / chat-assistant opening phrases that instruct models emit when
# they think they're replying to a user that flagged an error. These don't fit
# the single-stream self-revision we concatenate in Stage 1, so strip them.
_SYCOPHANTIC_PATTERNS = [
    # "You're absolutely right to point out/question/notice ...." -> drop whole sentence
    re.compile(r"^(?:You(?:'re| are))\s+(?:absolutely\s+)?right\s+to\s+[^.!?\n]*[.!?]\s*", re.IGNORECASE),
    # "You're absolutely right — " / "Yes, you are right, " / "Indeed, you're right: "
    re.compile(r"^(?:Yes[,!.]?\s+|Indeed[,!.]?\s+)?(?:You(?:'re| are))\s+(?:absolutely\s+)?right\s*[—\-,:.]+\s*", re.IGNORECASE),
    # "Thank you for ..."
    re.compile(r"^Thank\s+you\s+for[^.!?\n]*[.!?]\s*", re.IGNORECASE),
    # "Great question — ..." / "Great question."
    re.compile(r"^Great\s+question[^.!?\n]*?[.!?—\-]\s*", re.IGNORECASE),
    # "Good catch ..." / "Good point ..." / "Nice catch ..."
    re.compile(r"^(?:Good|Nice)\s+(?:catch|point|observation)[^.!?\n]*?[.!?—\-]\s*", re.IGNORECASE),
]


def clean_critique_response(response: str) -> str:
    """Strip sycophantic openings (e.g. "You're absolutely right — ") so the
    revised response reads as one continuous self-revision rather than a reply
    to a user. Safe on non-instruct outputs: when no pattern matches the text
    is returned unchanged.
    """
    if not response:
        return response

    text = response.lstrip()
    stripped = False
    changed = True
    while changed:
        changed = False
        for pat in _SYCOPHANTIC_PATTERNS:
            new_text, n = pat.subn("", text, count=1)
            if n > 0:
                text = new_text.lstrip()
                changed = True
                stripped = True

    # Re-capitalize only when a prefix was actually removed.
    if stripped and text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def remove_repeating_suffix(
    text: str,
    min_block: int = 10,
    max_block: int = 1500,
    min_repeats: int = 3,
    max_passes: int = 5,
) -> str:
    """Detect a periodic block that repeats at the end of `text` and trim down
    to a single copy. Handles the degenerate "loop until token budget"
    failure mode where the model emits the same chunk dozens of times.

    For each candidate period L (smallest first), extend the periodic region
    backward as far as possible (the invariant text[i] == text[i+L] holds).
    If the span covers at least `min_repeats` copies, keep one and drop the
    rest. Runs up to `max_passes` times because the trimmed tail can expose
    a different period at a different scale.
    """
    if not text:
        return text

    for _ in range(max_passes):
        n = len(text)
        if n < min_repeats * min_block:
            break
        trimmed = False
        for L in range(min_block, min(max_block, n // min_repeats) + 1):
            p = n - L
            while p > 0 and text[p - 1] == text[p - 1 + L]:
                p -= 1
            if n - p >= min_repeats * L:
                text = text[: p + L]
                trimmed = True
                break
        if not trimmed:
            break
    return text


def _clean_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """Trim degenerate repeating tails and strip sycophantic openings from the
    response fields of a D_revision tuple, in place."""
    if r.get("y_init"):
        r["y_init"] = remove_repeating_suffix(r["y_init"])
    if r.get("y_revised"):
        r["y_revised"] = clean_critique_response(remove_repeating_suffix(r["y_revised"]))
    return r


# --------------------------------------------------------------------------
# Loading / filtering
# --------------------------------------------------------------------------

def _load_records(input_path: str) -> List[Dict[str, Any]]:
    if input_path.endswith(".jsonl"):
        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_and_dedupe(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep tuples with a verified-correct revision; dedupe on
    (x, y_init, y_revised)."""
    out = []
    seen: set = set()
    for r in records:
        if not r.get("y_revised_correct", False):
            continue
        key = (r.get("x"), r.get("y_init"), r.get("y_revised"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _stage1_record(r: Dict[str, Any]) -> Dict[str, str]:
    """prompt = x; completion = y_init + P_r + y_revised."""
    completion = f"{r['y_init']}\n\n{r['p_r']}\n\n{r['y_revised']}"
    return {"prompt": r["x"], "completion": completion}


def _stage2_record(r: Dict[str, Any]) -> Dict[str, str]:
    """prompt = x + y_init + P_r; completion = y_revised."""
    prompt = f"{r['x']}\n\n{r['y_init']}\n\n{r['p_r']}"
    return {"prompt": prompt, "completion": r["y_revised"]}


def prepare_srt_data(
    input_path: str,
    output_dir: str,
    stage2_size: int = 1000,
    clean: bool = True,
    seed: int = 0,
    prefix: str = "srt",
) -> Tuple[str, str]:
    """Split D_revision into Stage 1 / Stage 2 SFT files.

    Stage 2 gets a fixed quota of `stage2_size` tuples; Stage 1 keeps *all*
    remaining incorrect-init tuples (its size is not fixed). This avoids the
    brittle "exact stage1_size + stage2_size must equal what was collected"
    split, since the number of verified revisions is uncertain.
    """
    records = _load_records(input_path)
    print(f"Loaded {len(records)} records from {input_path}")

    if clean:
        for r in records:
            _clean_record(r)
        print("Cleaned response fields (repeating-tail trim + sycophancy strip)")

    records = _filter_and_dedupe(records)
    print(f"After filter (y_revised_correct, dedupe): {len(records)}")
    if not records:
        raise ValueError("No usable revision tuples (y_revised_correct) after filtering.")

    # Stage 1 (generation loss) trains on the full mistake -> revision
    # trajectory as one stream, so it only uses incorrect-init tuples.
    # Stage 2 (revision loss) is a small fixed-size set that may mix both.
    incorrect_init = [r for r in records if not r.get("y_init_correct", False)]
    correct_init = [r for r in records if r.get("y_init_correct", False)]
    print(f"  incorrect-init: {len(incorrect_init)}  |  correct-init: {len(correct_init)}")

    rng = random.Random(seed)
    rng.shuffle(incorrect_init)
    rng.shuffle(correct_init)

    # Stage 2 takes up to `stage2_size` tuples. Correct-init tuples go first
    # (Stage 1 cannot use them anyway); the rest of the quota is filled with
    # incorrect-init. Stage 1 then takes ALL remaining incorrect-init, so the
    # bulk of the data lands in Stage 1.
    n_correct_s2 = min(len(correct_init), stage2_size)
    n_incorrect_s2 = min(len(incorrect_init), stage2_size - n_correct_s2)
    stage2_src = correct_init[:n_correct_s2] + incorrect_init[:n_incorrect_s2]
    rng.shuffle(stage2_src)

    stage1_src = incorrect_init[n_incorrect_s2:]  # all remaining incorrect-init

    if not stage1_src:
        raise ValueError(
            f"Stage 1 is empty: {len(incorrect_init)} incorrect-init tuples and "
            f"Stage 2 consumed {n_incorrect_s2} of them. Collect more data or lower "
            f"--stage2_size ({stage2_size})."
        )
    if not stage2_src:
        raise ValueError("Stage 2 is empty — no revision tuples available.")
    if len(stage1_src) < len(stage2_src):
        print(
            f"  [WARNING] Stage 1 ({len(stage1_src)}) is smaller than Stage 2 "
            f"({len(stage2_src)}); collect more incorrect-init data or lower --stage2_size."
        )
    unused_correct = len(correct_init) - n_correct_s2
    if unused_correct:
        print(
            f"  [note] {unused_correct} correct-init tuples unused (Stage 2 quota full; "
            f"Stage 1 takes incorrect-init only)."
        )

    stage1 = [_stage1_record(r) for r in stage1_src]
    stage2 = [_stage2_record(r) for r in stage2_src]

    os.makedirs(output_dir, exist_ok=True)
    s1_path = os.path.join(output_dir, f"{prefix}_stage1.json")
    s2_path = os.path.join(output_dir, f"{prefix}_stage2.json")

    with open(s1_path, "w", encoding="utf-8") as f:
        json.dump(stage1, f, indent=2, ensure_ascii=False)
    with open(s2_path, "w", encoding="utf-8") as f:
        json.dump(stage2, f, indent=2, ensure_ascii=False)

    print(f"\nStage 1 ({len(stage1)} records) -> {s1_path}")
    print(f"  prompt = x;  completion = y_init + P_r + y_revised  (incorrect-init only)")
    print(
        f"Stage 2 ({len(stage2)} records: {n_incorrect_s2} incorrect-init + "
        f"{n_correct_s2} correct-init) -> {s2_path}"
    )
    print(f"  prompt = x + y_init + P_r;  completion = y_revised")

    print("\n--- stage1 example ---")
    print("prompt:    ", stage1[0]["prompt"][:160].replace("\n", "\\n"))
    print("completion:", stage1[0]["completion"][:160].replace("\n", "\\n"))
    print("\n--- stage2 example ---")
    print("prompt:    ", stage2[0]["prompt"][:160].replace("\n", "\\n"))
    print("completion:", stage2[0]["completion"][:160].replace("\n", "\\n"))

    return s1_path, s2_path


def main():
    p = argparse.ArgumentParser(description="Build 2-stage SRT SFT data (Section 2.1)")
    p.add_argument(
        "--input",
        required=True,
        help="Path to D_revision JSON/JSONL produced by self_critique_pipeline.py",
    )
    p.add_argument(
        "--output_dir",
        default="train_data",
        help="Where to write stage1/stage2 JSON files",
    )
    p.add_argument(
        "--stage2_size",
        type=int,
        default=1000,
        help="Target size of the Stage 2 (revision-loss) set. Stage 1 "
        "(generation loss) takes ALL remaining incorrect-init tuples.",
    )
    p.add_argument(
        "--no_clean",
        action="store_true",
        help="Skip response cleaning (degenerate-tail trimming + sycophancy stripping)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="srt")
    args = p.parse_args()

    prepare_srt_data(
        input_path=args.input,
        output_dir=args.output_dir,
        stage2_size=args.stage2_size,
        clean=not args.no_clean,
        seed=args.seed,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
