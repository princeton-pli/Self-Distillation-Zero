import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, DatasetDict, Dataset
import transformers
import trl

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen3-4B-Instruct-2507")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="distillation")
    train_file_path: Optional[str] = field(default='open-r1')
    sample_size: Optional[int] = field(default=-1)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # Configure training args that do not depend on dataset/model
    args.completion_only_loss = True
    args.max_seq_length = config.block_size
    args.max_length = config.block_size
    args.use_liger_kernel = True

    dataset_name = None

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        if "Gemma" in config.model_name:
            model = transformers.Gemma3ForCausalLM.from_pretrained(config.model_name)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    # tokenizer setup (needed for dataset processing below)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    # Use model EOS token when available to keep stop tokens model-specific
    args.eos_token = tokenizer.eos_token or "<|im_end|>"
    if "Llama" in config.model_name:
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"
    elif "Gemma" in config.model_name:
        # Use a token that is never used
        tokenizer.pad_token = "<|pad|>"

    # Load dataset - support both HuggingFace datasets and local JSON files
    if config.train_file_path.endswith('.json') or config.train_file_path.endswith('.jsonl'):
        logging.info(f"Loading dataset from JSON file: {config.train_file_path}")
        # Check if it's a JSONL file (newline-delimited JSON)
        if config.train_file_path.endswith('.jsonl'):
            loaded_dataset = load_dataset("json", data_files=config.train_file_path)
            # Convert to DatasetDict format if needed
            if isinstance(loaded_dataset, DatasetDict):
                dataset = loaded_dataset
            else:
                dataset = DatasetDict({'train': loaded_dataset})
        else:
            # Regular JSON file - could be a list or a dict with train/test splits
            with open(config.train_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if dataset is in the new format (single dict with lists) or old format
            if isinstance(data, dict) and 'prompt' in data and 'completion' in data:
                # New format: single dictionary with lists
                if not isinstance(data['prompt'], list) or not isinstance(data['completion'], list):
                    raise ValueError(f"'prompt' and 'completion' must be lists. Got prompt: {type(data['prompt'])}, completion: {type(data['completion'])}")
                if len(data['prompt']) != len(data['completion']):
                    raise ValueError(f"'prompt' and 'completion' lists must have the same length. Got {len(data['prompt'])} and {len(data['completion'])}")
                
                # Detect format: text format (strings) or message format (list of dicts)
                if len(data['prompt']) > 0:
                    first_prompt = data['prompt'][0]
                    first_completion = data['completion'][0]
                    
                    # Check if it's text format (strings)
                    if isinstance(first_prompt, str) and isinstance(first_completion, str):
                        logging.info("Detected text format: prompt and completion are strings")
                        # Text format - use directly
                        dataset = DatasetDict({'train': Dataset.from_dict(data)})
                    # Check if it's message format (list of dicts)
                    elif isinstance(first_prompt, list) and isinstance(first_completion, list):
                        logging.info("Detected message format: prompt and completion are lists of message dicts")
                        # Validate message format
                        for i, prompt in enumerate(data['prompt']):
                            if not isinstance(prompt, list):
                                raise ValueError(f"Each element in 'prompt' must be a list of message dicts. Got type {type(prompt)} at index {i}")
                            for j, msg in enumerate(prompt):
                                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                                    raise ValueError(f"Each message in 'prompt' must be a dict with 'role' and 'content' keys. Got {type(msg)} at prompt[{i}][{j}]")
                        for i, completion in enumerate(data['completion']):
                            if not isinstance(completion, list):
                                raise ValueError(f"Each element in 'completion' must be a list of message dicts. Got type {type(completion)} at index {i}")
                            for j, msg in enumerate(completion):
                                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                                    raise ValueError(f"Each message in 'completion' must be a dict with 'role' and 'content' keys. Got {type(msg)} at completion[{i}][{j}]")
                        # Message format - use directly (SFTTrainer can handle this)
                        dataset = DatasetDict({'train': Dataset.from_dict(data)})
                    else:
                        raise ValueError(f"Unsupported format: prompt and completion must be either strings (text format) or lists of message dicts (message format). Got prompt: {type(first_prompt)}, completion: {type(first_completion)}")
                else:
                    # Empty dataset
                    dataset = DatasetDict({'train': Dataset.from_dict(data)})
            elif isinstance(data, dict) and ('train' in data or 'test' in data):
                # Dict with train/test splits
                dataset_dict = {}
                if 'train' in data:
                    # Check if train is in new format (dict with lists) or old format (list of dicts)
                    if isinstance(data['train'], dict) and 'prompt' in data['train'] and 'completion' in data['train']:
                        # New format: dict with prompt/completion lists
                        # Both text format (strings) and message format (list of dicts) are supported
                        dataset_dict['train'] = Dataset.from_dict(data['train'])
                    else:
                        # Old format: list of dicts, need to convert
                        if isinstance(data['train'], list) and len(data['train']) > 0:
                            if isinstance(data['train'][0], dict) and 'prompt' in data['train'][0] and 'completion' in data['train'][0]:
                                # Convert list of dicts to dict of lists
                                prompts = [ex['prompt'] for ex in data['train']]
                                completions = [ex['completion'] for ex in data['train']]
                                dataset_dict['train'] = Dataset.from_dict({'prompt': prompts, 'completion': completions})
                            else:
                                dataset_dict['train'] = Dataset.from_list(data['train'])
                        else:
                            dataset_dict['train'] = Dataset.from_list(data['train'])
                if 'test' in data:
                    if isinstance(data['test'], dict) and 'prompt' in data['test'] and 'completion' in data['test']:
                        # New format: dict with prompt/completion lists
                        # Both text format (strings) and message format (list of dicts) are supported
                        dataset_dict['test'] = Dataset.from_dict(data['test'])
                    else:
                        if isinstance(data['test'], list) and len(data['test']) > 0:
                            if isinstance(data['test'][0], dict) and 'prompt' in data['test'][0] and 'completion' in data['test'][0]:
                                prompts = [ex['prompt'] for ex in data['test']]
                                completions = [ex['completion'] for ex in data['test']]
                                dataset_dict['test'] = Dataset.from_dict({'prompt': prompts, 'completion': completions})
                            else:
                                dataset_dict['test'] = Dataset.from_list(data['test'])
                        else:
                            dataset_dict['test'] = Dataset.from_list(data['test'])
                dataset = DatasetDict(dataset_dict)
            elif isinstance(data, list):
                # Old format: list of examples
                if len(data) > 0 and isinstance(data[0], dict):
                    if 'prompt' in data[0] and 'completion' in data[0]:
                        # Convert list of dicts to dict of lists
                        # Supports both text format (strings) and message format (list of dicts)
                        prompts = [ex['prompt'] for ex in data]
                        completions = [ex['completion'] for ex in data]
                        dataset = DatasetDict({'train': Dataset.from_dict({'prompt': prompts, 'completion': completions})})
                    else:
                        raise ValueError(f"Examples must have 'prompt' and 'completion' fields. Found keys: {list(data[0].keys())}")
                else:
                    raise ValueError(f"Data must be a list of dictionaries. Got: {type(data[0]) if data else 'empty list'}")
            else:
                raise ValueError(f"JSON file must contain either a dict with 'prompt'/'completion' lists, a dict with 'train'/'test' keys, or a list of examples. Got: {type(data)}")
        
        # Validate dataset structure
        for split in dataset.keys():
            if 'prompt' not in dataset[split].column_names or 'completion' not in dataset[split].column_names:
                raise ValueError(f"Dataset split '{split}' must have 'prompt' and 'completion' columns. Found: {dataset[split].column_names}")
    else:
        # Load from allowed HuggingFace datasets by alias.
        dataset_aliases = {
            "open-r1": "data/openr1/openr1_math_220k.json",
        }
        dataset_name = config.train_file_path
        if dataset_name not in dataset_aliases:
            raise ValueError(
                f"Only {list(dataset_aliases)} are supported when using non-JSON datasets. "
                f"Got: {dataset_name}"
            )

        hf_dataset_name = dataset_aliases[dataset_name]
        logging.info(f"Loading dataset alias '{dataset_name}' -> '{hf_dataset_name}'")
        if hf_dataset_name.endswith('.json'):
            dataset = load_dataset("json", data_files=hf_dataset_name)
        else:
            dataset = load_dataset(hf_dataset_name)

        system_message = "Please reason step by step, and put your final answer within \\boxed{{}}."
        prompt_template = "{question}"

    if "r1" in config.train_file_path:
        system_message = "Please reason step by step, and put your final answer within \\boxed{{}}."
        prompt_template = (
            "{question}"
        )
    elif "code" in config.train_file_path:
        system_message = "You are a helpful assistant that solves coding problems step by step."
        prompt_template = (
            "{question}"
        )

    def convert_math_example(example):
        if "question" in example:
            prompt_key = "question"
        else:
            prompt_key = "prompt"
        
        if not prompt_template:
            user_prompt = example["prompt"]
        else:
            user_prompt = prompt_template.format(question=example[prompt_key])
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=None
            )
        except TypeError:
            # Older tokenizers may not support enable_thinking
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        completion = example["completion"] + "<|im_end|>"
        return {"prompt": prompt, "completion": completion}

    for split in dataset.keys():
        dataset[split] = dataset[split].map(
            convert_math_example,
            remove_columns=dataset[split].column_names
        )
    logging.info("Converted sft dataset to prompt/completion using chat template")

    # Optional subsampling for quick experiments
    if config.sample_size is not None and config.sample_size != -1:
        limit = min(config.sample_size, len(dataset['train']))
        dataset['train'] = dataset['train'].select(range(limit))
        logging.info(f"Subsampled train split to first {limit} examples (requested {config.sample_size})")

    print("DEBUG: training example:", dataset['train'][0])

    # setting up trainer
    trainer = trl.SFTTrainer(
        model,
        processing_class=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args
    )
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
