"""
Self-Revision Training (SRT) data collection pipeline — Phase 1 of SD-Zero.

Implements Section 2.1 of the paper. For each problem x with ground-truth a:

  1. Sample initial response   y_init   ~ pi_theta(. | x)
  2. Compute binary reward     r        = 1[answer(y_init) == a]
  3. Pick outcome-conditioned phrase P_r (model-specific; see _P_R_PHRASES):
         r = 1  ->  e.g. "Let me rephrase the above solution."
         r = 0  ->  e.g. "Wait, this response is wrong. Let me correct it."
  4. Sample revised response   y_revised ~ pi_theta(. | x, y_init, P_r)
  5. Keep tuples whose y_revised is verified correct.

The collected dataset is

    D_revision = { (x, y_init, P_r, y_revised) }

and is consumed by prepare_data.py to build the two SFT stages.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

math_verify_path = os.path.join(_here, "Math-Verify", "src")
if os.path.exists(math_verify_path):
    sys.path.insert(0, math_verify_path)

from math_evaluation.parser import extract_answer, strip_string
from math_evaluation.grader import math_equal

# Optional code judge — only needed in --mode code.
try:
    from code_runner.runner import judge_code_response
except Exception:  # noqa: BLE001
    judge_code_response = None


# Outcome-conditioned control phrases P_r (Section 2.1). The self-correction
# cue is matched to how each base model naturally phrases one, so it is
# selected per model family from the model path.
_P_R_PHRASES = {
    "qwen": {
        "correct": "Let me rephrase the above solution.",
        "incorrect": "Wait, this response is wrong. Let me correct it.",
    },
    "olmo": {
        "correct": "Let me rephrase the above solution.",
        "incorrect": "Wrong. I need to redo all.",
    },
}
_DEFAULT_P_R = _P_R_PHRASES["qwen"]


def select_p_r(model_path: str) -> Dict[str, str]:
    """Return the {correct, incorrect} P_r control phrases for the model family
    in `model_path` (substring match; falls back to the Qwen phrasing)."""
    name = str(model_path).lower()
    for family, phrases in _P_R_PHRASES.items():
        if family in name:
            return phrases
    return _DEFAULT_P_R


class SelfRevisionPipeline:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-4B-Instruct-2507",
        dataset_name: str = "open-r1/OpenR1-Math-220k",
        num_responses: int = 1,
        num_revisions: int = 1,
        batch_size: int = 32,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens: int = 16384,
        max_samples: Optional[int] = None,
        start_index: int = 0,
        end_index: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        use_math_verify: bool = False,
        target_revision_samples: Optional[int] = 6000,
        revise_correct: bool = False,
        subset: Optional[str] = None,
        mode: str = "math",
    ):
        if mode not in ("math", "code"):
            raise ValueError(f"mode must be 'math' or 'code', got {mode!r}")
        if mode == "code" and judge_code_response is None:
            raise ImportError(
                "mode=code requires code_runner.runner.judge_code_response to be importable"
            )
        self.mode = mode
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.num_responses = num_responses
        self.num_revisions = num_revisions
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_samples = max_samples
        self.start_index = start_index
        self.end_index = end_index
        self.use_math_verify = use_math_verify
        self.target_revision_samples = target_revision_samples
        self.revise_correct = revise_correct
        self.subset = subset
        self.p_r = select_p_r(model_path)

        print("Initializing SRT pipeline...")
        print(f"  Model: {model_path}")
        print(f"  Dataset: {dataset_name}{(' / ' + subset) if subset else ''}")
        print(f"  num_responses per x: {num_responses}")
        print(f"  num_revisions per y_init: {num_revisions}")
        print(f"  target D_revision size: {target_revision_samples}")
        print(f"  revise correct y_init too: {revise_correct}")
        print(f"  P_r (incorrect): {self.p_r['incorrect']!r}")

        print(f"\nLoading model from {model_path} ...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype="auto",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=32768,
        )

        print(f"\nLoading dataset {dataset_name} ...")
        if dataset_name.endswith((".json", ".jsonl")):
            if not os.path.exists(dataset_name):
                raise FileNotFoundError(dataset_name)
            self.dataset = load_dataset("json", data_files=dataset_name)
        elif subset:
            self.dataset = load_dataset(dataset_name, subset)
        else:
            self.dataset = load_dataset(dataset_name)

        total = len(self.dataset["train"])
        print(f"Loaded {total} examples")

        self.stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "Question:"]

    # ------------------------------------------------------------------ data

    def extract_problem_and_answer(self, example: Dict[str, Any]) -> Tuple[str, str]:
        if self.mode == "code":
            problem = example.get("prompt") or example.get("problem") or example.get("question")
            if problem is None:
                raise ValueError(f"No problem field in code example: {list(example.keys())}")
            # Reference solution only used for logging; correctness is judged by test cases.
            answer = ""
            for key in ("solution", "answer", "output", "generation", "target", "label"):
                if key in example and example[key]:
                    answer = str(example[key])
                    break
            return str(problem).strip(), answer.strip()

        problem = None
        for key in ("problem", "question", "query", "input", "text"):
            if key in example:
                problem = example[key]
                break
        if problem is None:
            raise ValueError(f"No problem field in example: {list(example.keys())}")

        answer = None
        for key in ("answer", "solution", "target", "label", "output"):
            if key in example:
                answer = example[key]
                break
        if answer is None:
            raise ValueError(f"No answer field in example: {list(example.keys())}")

        if isinstance(answer, str) and "####" in answer:
            answer = answer.split("####")[1].strip()
        return str(problem).strip(), str(answer).strip()

    # ----------------------------------------------------------------- prompts

    def _system_msg(self) -> str:
        if self.mode == "code":
            return "You are a helpful assistant that solves coding problems step by step."
        return "You are a helpful assistant that solves math problems step by step."

    def _user_suffix(self) -> str:
        if self.mode == "code":
            return ""
        return "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

    def create_initial_prompt(self, problem: str) -> str:
        """Chat-format prompt for sampling y_init."""
        return (
            "<|im_start|>system\n"
            f"{self._system_msg()}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{problem}{self._user_suffix()}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def create_revision_prompt(self, problem: str, y_init: str, p_r: str) -> str:
        """
        Continuation prompt for sampling y_revised ~ pi(. | x, y_init, P_r).

        y_init lives inside the assistant turn, followed by the control phrase
        P_r; the model continues from there.
        """
        return (
            "<|im_start|>system\n"
            f"{self._system_msg()}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{problem}{self._user_suffix()}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{y_init}\n\n{p_r}\n\n"
        )

    # ----------------------------------------------------------------- judging

    def judge_response(
        self,
        response: str,
        ground_truth: str,
        example: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        if self.mode == "code":
            is_correct, extracted, _ = judge_code_response(response, example or {})
            return bool(is_correct), str(extracted)

        extracted = ""
        try:
            extracted = strip_string(extract_answer(response, "math"))
        except Exception:
            pass

        if self.use_math_verify:
            try:
                from math_verify import parse, verify
                gold = parse(ground_truth)
                pred = parse(response)
                return bool(verify(gold, pred)), str(pred)
            except Exception as e:
                print(f"math_verify failed ({e}); falling back to math_equal")

        try:
            if not extracted:
                extracted = strip_string(extract_answer(response, "math"))
            return bool(math_equal(extracted, strip_string(ground_truth), timeout=False)), extracted
        except Exception as e:
            print(f"judge_response error: {e}")
            return False, extracted

    # ----------------------------------------------------------------- batching

    def _generate(self, prompts: List[str], n: int) -> List[List[str]]:
        params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=n,
            stop=self.stop_tokens,
        )
        outputs = self.llm.generate(prompts, params)
        return [[o.text for o in obj.outputs] for obj in outputs]

    # ------------------------------------------------------------------- main

    def run(self, output_path: str) -> List[Dict[str, Any]]:
        """Collect D_revision and write to output_path."""
        print("\n" + "=" * 80)
        print("SRT data collection")
        print("=" * 80)
        print(f"timestamp: {datetime.now().isoformat()}")

        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        all_responses_path = os.path.splitext(output_path)[0] + "_all_responses.jsonl"
        if os.path.exists(all_responses_path):
            os.remove(all_responses_path)

        d_revision: List[Dict[str, Any]] = []
        split = self.dataset["train"]
        total = len(split)
        start = max(0, min(self.start_index, total))
        end = total if self.end_index is None else max(start, min(self.end_index, total))
        if self.max_samples is not None:
            end = min(end, start + self.max_samples)
        print(f"Scanning examples [{start}, {end})")

        n_init_total = 0
        n_init_correct = 0
        n_revised_total = 0
        n_revised_correct = 0

        def _serializable(rec: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in rec.items() if k != "example"}

        pbar = tqdm(range(start, end, self.batch_size), desc="batches")
        for batch_start in pbar:
            batch_end = min(batch_start + self.batch_size, end)

            items: List[Dict[str, Any]] = []
            for i in range(batch_start, batch_end):
                ex = split[i]
                try:
                    problem, answer = self.extract_problem_and_answer(ex)
                except Exception as e:
                    print(f"skip {i}: {e}")
                    continue
                items.append({"index": i, "x": problem, "a": answer, "example": ex})
            if not items:
                continue

            # 1. y_init
            init_prompts = [self.create_initial_prompt(it["x"]) for it in items]
            init_completions = self._generate(init_prompts, n=self.num_responses)

            # 2-3. judge + assign P_r; build revision prompts
            revision_inputs: List[Dict[str, Any]] = []
            for it, completions in zip(items, init_completions):
                for resp_idx, y_init in enumerate(completions):
                    is_correct, extracted = self.judge_response(
                        y_init, it["a"], example=it["example"]
                    )
                    n_init_total += 1
                    if is_correct:
                        n_init_correct += 1
                    if is_correct and not self.revise_correct:
                        continue
                    p_r = self.p_r["correct"] if is_correct else self.p_r["incorrect"]
                    revision_inputs.append(
                        {
                            "x": it["x"],
                            "a": it["a"],
                            "example": it["example"],
                            "y_init": y_init,
                            "y_init_correct": is_correct,
                            "y_init_extracted": extracted,
                            "p_r": p_r,
                            "original_index": it["index"],
                            "response_idx": resp_idx,
                        }
                    )

            # 4. y_revised
            if revision_inputs:
                rev_prompts = [
                    self.create_revision_prompt(r["x"], r["y_init"], r["p_r"])
                    for r in revision_inputs
                ]
                rev_completions = self._generate(rev_prompts, n=self.num_revisions)

                # 5. filter to verified-correct
                for r, completions in zip(revision_inputs, rev_completions):
                    for rev_idx, y_revised in enumerate(completions):
                        rev_correct, rev_extracted = self.judge_response(
                            y_revised, r["a"], example=r["example"]
                        )
                        n_revised_total += 1
                        if rev_correct:
                            n_revised_correct += 1

                        record = {
                            "x": r["x"],
                            "y_init": r["y_init"],
                            "p_r": r["p_r"],
                            "y_revised": y_revised,
                            "ground_truth": r["a"],
                            "y_init_correct": r["y_init_correct"],
                            "y_revised_correct": rev_correct,
                            "y_init_extracted": r["y_init_extracted"],
                            "y_revised_extracted": rev_extracted,
                            "original_index": r["original_index"],
                            "response_idx": r["response_idx"],
                            "revision_idx": rev_idx,
                        }

                        with open(all_responses_path, "a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(_serializable(record), ensure_ascii=False) + "\n"
                            )

                        if rev_correct:
                            d_revision.append(record)

            pbar.set_postfix(
                kept=len(d_revision),
                init_acc=f"{(n_init_correct/n_init_total*100 if n_init_total else 0):.1f}%",
                rev_acc=f"{(n_revised_correct/n_revised_total*100 if n_revised_total else 0):.1f}%",
            )

            if (
                self.target_revision_samples is not None
                and len(d_revision) >= self.target_revision_samples
            ):
                print(f"\nReached target ({self.target_revision_samples}); stopping early.")
                break

        if (
            self.target_revision_samples is not None
            and len(d_revision) > self.target_revision_samples
        ):
            d_revision = d_revision[: self.target_revision_samples]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [_serializable(r) for r in d_revision], f, indent=2, ensure_ascii=False
            )

        summary = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "dataset_name": self.dataset_name,
            "num_responses_per_x": self.num_responses,
            "num_revisions_per_y_init": self.num_revisions,
            "revise_correct": self.revise_correct,
            "scanned_range": [start, end],
            "n_init_total": n_init_total,
            "n_init_correct": n_init_correct,
            "n_revised_total": n_revised_total,
            "n_revised_correct": n_revised_correct,
            "d_revision_size": len(d_revision),
            "output_file": output_path,
            "all_responses_file": all_responses_path,
        }
        with open(
            os.path.splitext(output_path)[0] + "_summary.json", "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print(f"Collected |D_revision| = {len(d_revision)}")
        print(f"y_init accuracy:    {n_init_correct}/{n_init_total}")
        print(f"y_revised accuracy: {n_revised_correct}/{n_revised_total}")
        print(f"Saved to {output_path}")
        return d_revision


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SRT data collection (paper Section 2.1)")
    parser.add_argument("--mode", choices=["math", "code"], default="math",
                        help="Task family: math (boxed-answer grading) or code (test cases)")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset", type=str, default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--num_responses", type=int, default=1)
    parser.add_argument("--num_revisions", "--num_critiques", dest="num_revisions",
                        type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--output", type=str, default="d_revision.json")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--math_verify", action="store_true")
    parser.add_argument("--target_revision_samples", type=int, default=6000,
                        help="Stop once this many verified-correct revisions collected")
    parser.add_argument("--revise_correct", action="store_true",
                        help="Also generate y_revised for correct y_init (paper r=1 path)")

    args = parser.parse_args()

    pipeline = SelfRevisionPipeline(
        model_path=args.model_path,
        dataset_name=args.dataset,
        num_responses=args.num_responses,
        num_revisions=args.num_revisions,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_samples=args.max_samples,
        start_index=args.start_index,
        end_index=args.end_index,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_math_verify=args.math_verify,
        target_revision_samples=args.target_revision_samples,
        revise_correct=args.revise_correct,
        subset=args.subset,
        mode=args.mode,
    )
    pipeline.run(output_path=args.output)


if __name__ == "__main__":
    main()
