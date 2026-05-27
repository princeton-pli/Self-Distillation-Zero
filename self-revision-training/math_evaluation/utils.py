import os
import json
import random
import json
from datetime import datetime
import os
import re
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any

try:
    from examples import get_examples, get_skill_examples, get_random_examples
except ImportError:
    from math_evaluation.examples import get_examples, get_skill_examples, get_random_examples


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


EXAMPLES = get_examples()

# Ours
# def extract_skills(output):
#     match = re.search(r"<skill>(.*?)</skill>", output, flags=re.DOTALL)
#     if not match:
#         return []  # No <skill>...</skill> block found

#     # Get the contents inside the <skill>...</skill> tags
#     skill_content = match.group(1)

#     # Split by comma, strip whitespace
#     skills = [skill.strip() for skill in skill_content.split(",") if skill.strip()]

#     return skills

def retrieve_skills(data_name, question, subject, split="test"):
    if data_name == "math-skill" and subject:
        subject_path = "_".join(subject.split(" "))
        input_dir = f"../skill_identifier/skill_retrieval/MATH/{subject_path}/{split}"
    elif data_name == "gsm8k-skill":
        input_dir = f"../skill_identifier/skill_retrieval/GSM8K/{split}"
    input_path = os.path.join(input_dir, f"{split}.json")
    with open(input_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    
    if question not in data["question2skill"]:
        return None
    skill_list = data["question2skill"][question]

    return skill_list


def load_prompt(example, data_name, prompt_type, num_shots, num_skill_shots, random_shots=False, llm_sol=False):
    if not num_shots:
        return []

    if data_name in ["math", "gsm8k", "math-skil", "gsm8k-skill"] and random_shots:
        return get_random_examples(data_name)[data_name][:num_shots]

    if data_name in ["math-skill", "gsm8k-skill"]:
        shot_examples = []

        # Get subject
        if data_name == "math-skill":
            subject = example["subject"].lower()
            subject = subject.replace("&", "and")
        else:
            subject = None
        
        # Get skills
        if "parsed_output" not in example: # retrieved_skills
            skills = retrieve_skills(data_name, example["question"], subject)
        else: # feedback_skills
            skills = example["parsed_output"]["skills"]
        # else:
        #     skills = example["missing_skills"].strip().split(", ")
        
        skills = skills[:num_skill_shots]
        if skills:
            for _ in range(num_skill_shots - len(skills)):
                skill_examples = get_skill_examples(skills[0], data_name, subject)
                if skill_examples:
                    skill_examples = skill_examples[data_name]
                    random.seed(datetime.now().timestamp())
                    random.shuffle(skill_examples)
                    shot_examples.append(skill_examples[0])
            for skill in skills:
                skill_examples = get_skill_examples(skill, data_name, subject)
                if skill_examples:
                    skill_examples = skill_examples[data_name]
                    random.seed(datetime.now().timestamp())
                    random.shuffle(skill_examples)
                    shot_examples.append(skill_examples[0])
        if len(shot_examples) < num_shots:
            if data_name == "math-skill":
                shot_examples.extend(EXAMPLES["math"][:(num_shots-len(shot_examples))])
            elif data_name == "gsm8k-skill":
                shot_examples.extend(EXAMPLES["gsm8k"][:(num_shots-len(shot_examples))])
        else:
            shot_examples = shot_examples[:num_shots]
        return shot_examples


    if data_name in ["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"]:
        data_name = "gsm8k"
    if data_name in ["math_oai", "hungarian_exam", "math-oai", "aime24", "amc23"]:
        data_name = "math"
    if data_name in ["sat_math"]:
        data_name = "mmlu_stem"
    if data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        data_name = "gaokao"

    # Ours
    if data_name in ["math", "gsm8k"] and llm_sol:
        data_name = data_name + "-llm"
    

    if prompt_type in ["tool-integrated"]:
        prompt_type = "tora"

    return EXAMPLES[data_name][:num_shots]


PROMPT_TEMPLATES = {
    "direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
    "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    "pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
    "tool-integrated": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
    "self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "wizard_zs": (
        "### Instruction:\n{input}\n\n### Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "platypus_fs": (
        "### Instruction:\n{input}\n\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "ap_fs": (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{input}\n\n### Assistant\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "deepseek-math": (
        "User: {input}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "kpmath": (
        "User: Please reason step by step and put your final answer at the end "
        'with "The answer is: ".\n\n{input}\n\nAssistant:',
        "{output}",
    ),
    "jiuzhang": (
        "## Question\n{input}\n\n## Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_tora": (
        "## Question\n{input}\n\n## Code Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_nl": (
        "## Question\n{input}\n\n## Natural Language Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "mmiqc": (
        'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
        "{output}",
        "\n\n\n",
    ),
    "abel": (
        "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
        "{output}",
        "\n\n",
    ),
    "shepherd": ("{input}\n", "{output}", "\n\n\n"),
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen3": (
        # "<|im_start|>system\n"
        # "Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen3_think": (
        # "<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n{input}<|im_end|>\n"
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n<think>",
        "{output}",
        "\n\n",
    ),
    "qwen3_no_think": (
        # "<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n{input}<|im_end|>\n"
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\nno_think",
        "{output}",
        "\n\n",
    ),
    "deepseek_think": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n",
        "{output}",
        "\n\n",
    ),
    "mathstral": (
        "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "{output}",
        "\n\n",
    ),
    "internlm-math-fs": ("Question:{input}\nAnswer:", "{output}", "\n"),
    "internlm-math-chat": (
        "<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mistral": (
        "[INST] {input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "numina": ("### Problem: {input}\n### Solution:", " {output}", "\n\n")
}

# Ours: to edit the code minimally, we use args.num_shots>100 to represent skill-enhance context, actual num_shots = args.num_shots-100


def construct_prompt(example, data_name, args):
    if args.adapt_few_shot and data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        demos = load_prompt(data_name, args.prompt_type, 5, args.num_skill_shots, args.random_shots, args.llm_sol)
    else:
        demos = load_prompt(example, data_name, args.prompt_type, args.num_shots, args.num_skill_shots, args.random_shots, args.llm_sol)
        if "missing_skills" in example and example["parsed_output"]:
            feedback = example["parsed_output"]
        else:
            feedback = ""
    prompt_type = args.prompt_type
    if prompt_type == "platypus_fs":
        prompt_type = "cot"
    if prompt_type == "tool-integrated":
        prompt_type = "tora"

    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]

    splitter = prompt_temp[2]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
    if args.prompt_type == "qwen25-math-cot":
        # Hotfix to support putting all demos into a single turn
        demo_prompt = splitter.join(["Question: " + q + "\n\n" + "Response: " + a for q, a in demos])
    else:
        demo_prompt = splitter.join(
            [
                input_template.format(input=q) + output_template.format(output=a)
                for q, a in demos
            ]
        )
    context = input_template.format(input=example["question"])
    if len(demo_prompt) == 0 and (
        args.adapt_few_shot and example["gt_ans"] not in ["A", "B", "C", "D", "E"]
    ):
        full_prompt = context
    else:
        if args.prompt_type == "qwen25-math-cot":
            # Hotfix to supportting put all demos into a single turn
            full_prompt = demo_prompt + splitter + "Question: " + example["question"]
            full_prompt = input_template.format(input=full_prompt)
        else:
            full_prompt = demo_prompt + splitter + context

    if args.prompt_type == "platypus_fs":
        full_prompt_temp = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        full_prompt = full_prompt_temp.format(instruction=full_prompt)

    if prompt_type == "tora":
        full_prompt = (
            """Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a `\boxed{}` without any units.
- Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

Here are some examples you may refer to:

---

"""
            + full_prompt
        )

    return full_prompt.strip(" ")  # important!


key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}


def show_sample(sample, print_all_preds=False):
    print("==" * 20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample["question"]))
    if "code" in sample:
        if print_all_preds:
            for code in sample["code"]:
                print("-" * 20)
                print("code:", code)
            print("Execution:", sample["report"])
        else:
            print("Solution:\n", sample["code"][0])
            print("Execution:", sample["report"][0])
    if "pred" in sample:
        print("Prediction:", repr(sample["pred"][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()
