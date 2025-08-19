# train_grpo.py
#
# See https://github.com/willccbb/verifiers for ongoing developments
#

import os
import re
import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def train_model_from_scratch(model, tokenizer, output_dir=None, per_device_train_batch_size: int = 32):
    dataset = get_gsm8k_questions()
    
    if model_name is None:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    if output_dir is None:
        if "Llama" in model_name:
            output_dir = "outputs/Llama-1B-GRPO"
            run_name = "Llama-1B-GRPO-gsm8k"
        else:
            output_dir="outputs/Qwen-1.5B-GRPO"
            run_name="Qwen-1.5B-GRPO-gsm8k"
    else:
        run_name = os.path.basename(output_dir)
        
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=per_device_train_batch_size, # in h100 
        gradient_accumulation_steps=8,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=250,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


def run_grpo_rollout_on_gsm8k(
    model, 
    tokenizer,
    num_generations: int = 4,
    per_device_batch_size: int = 1,
    steps_per_generation: int = 2,
    max_completion_length: int = 250,
    temperature: float = 0.8,
    max_batches: int = 10,
    verbose: bool = True, 
    output_dir: str = None
):
    """
    Run GRPO rollout demo on GSM8K test dataset.
    
    Args:
        model_name: HuggingFace model name to load
        checkpoint_dir: Optional checkpoint directory to load model from
        num_generations: Number of completions per prompt
        per_device_batch_size: Batch size per device
        steps_per_generation: Number of steps per generation
        max_completion_length: Maximum length of generated completions
        temperature: Sampling temperature
        max_batches: Maximum number of batches to process
        verbose: Whether to print detailed information
    """
    from generate_rollouts import grpo_rollout_demo, simple_length_reward
    
    print("🚀 Running GRPO Rollout Demo on GSM8K Test Dataset")
    
    # Load GSM8K test dataset
    print("Loading GSM8K test dataset...")
    gsm8k_dataset = get_gsm8k_questions(split="test")
    
    # Convert GSM8K format to chat format for rollout demo
    def convert_gsm8k_to_rollout_format(examples):
        prompts = []
        answers = []
        for example in examples:
            # Keep the chat format instead of flattening
            prompts.append(example["prompt"])
            answers.append(example["answer"])
        
        return {"prompt": prompts, "answer": answers}
    
    # Convert dataset format
    rollout_dataset = Dataset.from_dict(
        convert_gsm8k_to_rollout_format(gsm8k_dataset)
    )
    
    print(f"Converted {len(rollout_dataset)} GSM8K examples for rollout demo")
    
    # Run the rollout demo
    results = grpo_rollout_demo(
        dataset=rollout_dataset,
        model=model,
        tokenizer=tokenizer,
        prompt_column="prompt",
        num_generations=num_generations,
        per_device_batch_size=per_device_batch_size,
        steps_per_generation=steps_per_generation,
        max_completion_length=max_completion_length,
        temperature=temperature,
        max_batches=max_batches,
        verbose=verbose,
        log_file=output_dir,
        log_format="jsonl",
        reward_funcs=[correctness_reward_func, soft_format_reward_func],
        reward_processing_classes=None  # Custom functions don't need tokenizers
    )
    
    print(f"\n✅ GSM8K GRPO Rollout Complete!")
    print(f"📊 Summary:")
    print(f"  • Processed {results['stats']['total_unique_prompts']} unique prompts")
    print(f"  • Generated {results['stats']['total_completions_generated']} total completions")
    print(f"  • Average completion length: {results['stats']['avg_completion_length']:.1f} words")
    
    if results['reward_scores']:
        print(f"\n🏆 Average Reward Scores:")
        for func_name, stats in results['stats']['mean_rewards_per_function'].items():
            print(f"  • {func_name}: {stats['mean']:.3f} (min: {stats['min']:.3f}, max: {stats['max']:.3f})")
    
    print(f"\n📝 Detailed results saved to: {results['log_file_path']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate GRPO model on GSM8K")
    parser.add_argument("--mode", choices=["train", "eval", "rollout"], required=True,
                        help="Whether to train a new model, evaluate from checkpoint, or run rollout demo")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to checkpoint directory for evaluation/rollout mode")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model name for training/rollout mode")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory for training mode")
    parser.add_argument("--max_batches", type=int, default=10,
                        help="Maximum number of batches for rollout mode")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of generations per prompt for rollout mode")
    parser.add_argument("--per_device_batch_size", type=int, default=4,
                        help="Number of per device batch size")
    
    args = parser.parse_args()
    
        
    # Load model and tokenizer
    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        print(f"Loading model from checkpoint: {args.checkpoint_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=None
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    else:
        print(f"Loading model: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=None
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "train":
        print("Starting training mode...")
        train_model_from_scratch(model, tokenizer, output_dir=args.output_dir)
    elif args.mode == "rollout":
        print("Starting GRPO rollout demo mode...")
        run_grpo_rollout_on_gsm8k(
            model, 
            checkpoint_dir=args.checkpoint,
            max_batches=args.max_batches,
            num_generations=args.num_generations,
            verbose=True, 
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()