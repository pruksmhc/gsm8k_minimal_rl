#!/usr/bin/env python3
"""
Utils function for rolling out with any HuggingFace dataset,
model, and tokenizer. Good for evaluating at each checkpoint, the diversity of generations.
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    set_seed
)
from datasets import Dataset
from trl.trainer.grpo_trainer import RepeatSampler
from trl.data_utils import maybe_apply_chat_template, is_conversational

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import json
import os
from datetime import datetime
import torch.nn as nn

RewardFunc = Union[str, nn.Module, Callable[[List[str], List[str]], List[float]]]

def _setup_logging(
    log_file: Optional[str], 
    log_format: str, 
    model: AutoModelForCausalLM, 
    verbose: bool
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Setup logging file and return log file path and log data structure."""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = getattr(model.config, '_name_or_path', 'unknown_model').split('/')[-1]
        log_file = f"rollout_completions_{model_name}_{timestamp}.{log_format}"
    
    log_file_path = os.path.abspath(log_file)
    
    if verbose:
        print(f"📝 Logging completions to: {log_file_path}")
        print(f"Log format: {log_format}")
    
    # Validate log format
    if log_format not in ["jsonl", "json", "txt"]:
        raise ValueError(f"log_format must be one of ['jsonl', 'json', 'txt'], got '{log_format}'")
    
    # Initialize log file
    log_data = None
    if log_format == "json":
        log_data = {
            "metadata": {
                "model_name": getattr(model.config, '_name_or_path', 'unknown_model'),
                "timestamp": datetime.now().isoformat(),
                "num_generations": None,  # Will be set by caller
                "temperature": None,      # Will be set by caller
                "top_p": None,           # Will be set by caller
                "max_completion_length": None,  # Will be set by caller
                "seed": None             # Will be set by caller
            },
            "completions": []
        }
    elif log_format == "txt":
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {getattr(model.config, '_name_or_path', 'unknown_model')}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
    
    return log_file_path, log_data


def _setup_model_and_tokenizer(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    seed: int, 
    verbose: bool
) -> None:
    """Setup model and tokenizer configurations."""
    # Set seed for reproducibility
    set_seed(seed)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if verbose:
            print(f"Set pad_token to eos_token: {tokenizer.eos_token}")


def _create_sampler_and_dataloader(
    dataset: Dataset, 
    num_generations: int, 
    per_device_batch_size: int, 
    steps_per_generation: int, 
    seed: int, 
    verbose: bool
) -> DataLoader:
    """Create RepeatSampler and DataLoader for rollout."""
    sampler = RepeatSampler(
        data_source=dataset,
        mini_repeat_count=num_generations,      # Repeat each prompt num_generations times
        batch_size=per_device_batch_size,       # Unique prompts per batch
        repeat_count=steps_per_generation,      # Repeat the whole process
        shuffle=True,
        seed=seed
    )
    
    if verbose:
        print(f"RepeatSampler will produce {len(sampler)} total samples")
    
    generation_batch_size = per_device_batch_size * steps_per_generation
    dataloader = DataLoader(
        dataset,
        batch_size=generation_batch_size * num_generations,  
        sampler=sampler,
        collate_fn=lambda x: x  # No collation, keep as list
    )
    
    return dataloader


def _process_batch(
    batch: List[Dict[str, Any]], 
    batch_idx: int, 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    generation_config: GenerationConfig, 
    processed_reward_funcs: List[RewardFunc], 
    reward_func_names: List[str], 
    prompt_column: str, 
    prompt_id_column: str, 
    log_file_path: str, 
    log_format: str, 
    log_data: Optional[Dict[str, Any]], 
    verbose: bool
) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]], Dict[str, Dict[str, float]]]:
    """Process a single batch of prompts and generate completions."""
    if verbose:
        print(f"\n--- Batch {batch_idx + 1} ---")
    
    # Extract prompts from the specified column
    maybe_conversational_prompts = [item[prompt_column] for item in batch]
    prompt_ids = [item[prompt_id_column] for item in batch]
    if verbose:
        print(f"Batch size: {len(maybe_conversational_prompts)}")
    
    # Analyze the duplication pattern
    unique_prompts = list(dict.fromkeys(prompt_ids))  # Preserve order, remove duplicates
    
    if verbose:
        print(f"Unique prompts in batch: {len(unique_prompts)}")
        for i, prompt in enumerate(unique_prompts):
            count = prompt_ids.count(prompt)
            print(f"  {i+1}. '{prompt[:50]}...' (repeated {count} times)")

    # Handle both string prompts and chat format using TRL utilities
    # Check if any prompt in the batch is conversational
    sample_item = batch[0] if batch else {}
    if is_conversational(sample_item):
        # Apply chat template to each prompt
        formatted_prompts = []
        for item in batch:
            formatted = maybe_apply_chat_template(item, tokenizer)
            formatted_prompts.append(formatted["prompt"])
        
        prompt_inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left", 
            truncation=True,
            add_special_tokens=False
        )
    else:
        # String format - tokenize directly
        prompt_inputs = tokenizer(
            maybe_conversational_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left", 
            truncation=True,
            add_special_tokens=False
        )
    
    # Move inputs to same device as model
    device = next(model.parameters()).device
    prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
    
    if verbose:
        print(f"Tokenized input shape: {prompt_inputs['input_ids'].shape}")
    
    # Generate completions (single call for all duplicated prompts)
    if verbose:
        print("🔄 Generating completions...")
    
    generated_ids = model.generate(
        **prompt_inputs,
        generation_config=generation_config,
        disable_compile=True
    )
    
    # Extract completion parts
    prompt_length = prompt_inputs['input_ids'].size(1)
    completion_ids = generated_ids[:, prompt_length:]
    
    if verbose:
        print(f"Generated completions shape: {completion_ids.shape}")
    
    # Decode completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    if is_conversational(sample_item):
        completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]
    else:
        completions = completions_text

    # Group completions by unique prompt
    batch_completion_groups = {}
    batch_reward_scores = {}
    
    for unique_prompt in unique_prompts:
        prompt_completions = []
        batch_examples = []  # Store full dataset examples for this prompt
        for j, prompt in enumerate(prompt_ids):
            if prompt == unique_prompt:
                prompt_completions.append(completions[j])
                batch_examples.append(batch[j])  # Store the full example from the dataset
                maybe_templated_prompt = maybe_conversational_prompts[j]
        batch_completion_groups[unique_prompt] = prompt_completions
        
        # Calculate rewards if reward functions are provided
        if processed_reward_funcs:
            prompt_rewards = _calculate_rewards(
                maybe_templated_prompt,
                prompt_completions, 
                processed_reward_funcs, 
                reward_func_names,
                inputs=batch_examples,  # Pass the full dataset examples
                verbose=verbose
            )
            batch_reward_scores[unique_prompt] = prompt_rewards
        
        # Log completions to file (now includes rewards if available)
        _log_completions_to_file(
            unique_prompt, 
            prompt_completions, 
            log_file_path, 
            log_format, 
            batch_idx,
            log_data if log_format == "json" else None,
            batch_reward_scores.get(unique_prompt, {})
        )
    
    if verbose:
        # Show results for this batch
        print("\n📝 Generated Completions:")
        for i, unique_prompt in enumerate(unique_prompts):
            print(f"\n🎯 Prompt {i+1}: '{unique_prompt[:70]}...'")
            
            prompt_completions = batch_completion_groups[unique_prompt]
            for k, completion in enumerate(prompt_completions):
                print(f"   Completion {k+1}: '{completion[:100]}...'")
    
    return (maybe_conversational_prompts, unique_prompts, completions, 
            batch_completion_groups, batch_reward_scores)


def _finalize_stats_and_logging(
    all_prompts: List[str], 
    all_completions: List[str], 
    completion_groups: Dict[str, List[str]], 
    reward_scores: Dict[str, Dict[str, List[float]]], 
    reward_func_names: List[str], 
    reward_funcs: Optional[List[RewardFunc]],
    batch_stats: List[Dict[str, Any]], 
    log_file_path: str, 
    log_format: str, 
    log_data: Optional[Dict[str, Any]], 
    verbose: bool
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Calculate final statistics and finalize logging."""
    # Calculate mean rewards per unique prompt
    mean_reward_scores = {}
    if reward_scores:
        for prompt, func_scores in reward_scores.items():
            mean_reward_scores[prompt] = {}
            for func_name, scores in func_scores.items():
                if scores:  # Only if we have scores
                    mean_reward_scores[prompt][func_name] = sum(scores) / len(scores)
                else:
                    mean_reward_scores[prompt][func_name] = None
    
    # Finalize logging
    if log_format == "json":
        # Add reward summary to JSON log
        if mean_reward_scores:
            log_data["reward_summary"] = mean_reward_scores
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"✅ Completions logged to: {log_file_path}")
    
    # Calculate overall statistics
    if all_completions:
        all_completion_lengths = [len(comp.split()) for comp in all_completions]
        overall_stats = {
            'total_prompts_processed': len(all_prompts),
            'total_unique_prompts': len(completion_groups),
            'total_completions_generated': len(all_completions),
            'avg_completion_length': sum(all_completion_lengths) / len(all_completion_lengths),
            'min_completion_length': min(all_completion_lengths),
            'max_completion_length': max(all_completion_lengths),
            'batches_processed': len(batch_stats),
            'log_file_path': log_file_path,
            'reward_functions_used': reward_func_names if reward_funcs else []
        }
        
        # Add reward statistics
        if mean_reward_scores:
            overall_stats['mean_rewards_per_function'] = {}
            for func_name in reward_func_names:
                all_scores = [scores[func_name] for scores in mean_reward_scores.values() if scores[func_name] is not None]
                if all_scores:
                    overall_stats['mean_rewards_per_function'][func_name] = {
                        'mean': sum(all_scores) / len(all_scores),
                        'min': min(all_scores),
                        'max': max(all_scores)
                    }
    else:
        overall_stats = {'log_file_path': log_file_path, 'reward_functions_used': reward_func_names if reward_funcs else []}
    
    return overall_stats, mean_reward_scores


def rollout_eval(
    dataset: Dataset,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_column: str = "prompt",
    prompt_id_column: str = "question",  # Can be same as prompt_column, is the un-templated prompt before conversation templating. 
    num_generations: int = 4,
    per_device_batch_size: int = 1,
    steps_per_generation: int = 2,
    max_completion_length: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int = 42,
    max_batches: int = 1,
    verbose: bool = True,
    log_file: Optional[str] = None,
    log_format: str = "jsonl",
    reward_funcs: Optional[Union[RewardFunc, List[RewardFunc]]] = None,
) -> Dict[str, Any]:
    """
    
    Args:
        dataset: HuggingFace Dataset containing prompts
        model: Pre-trained causal language model
        tokenizer: Tokenizer corresponding to the model
        prompt_column: Column name containing prompts in the dataset
        num_generations: Number of completions per prompt 
        per_device_batch_size: Batch size per device 
        steps_per_generation: Number of training steps per generation 
        max_completion_length: Maximum length of generated completions
        temperature: Sampling temperature for generation
        top_p: Nucleus sampling parameter
        seed: Random seed for reproducibility
        max_batches: Maximum number of batches to process (for demo purposes)
        verbose: Whether to print detailed information
        log_file: Path to file for logging completions. If None, auto-generates filename
        log_format: Format for logging ("jsonl", "json", or "txt")
        reward_funcs: Optional reward functions (str model path, nn.Module, or callable)
        
    Returns:
        Dict containing:
        - prompts: List of all prompts processed
        - completions: List of all generated completions
        - completion_groups: Dict mapping prompt to its completions
        - reward_scores: Dict mapping prompt to mean reward scores per function
        - stats: Generation statistics
        - log_file_path: Path to the generated log file
    """
    
    if verbose:
        print(f"Dataset size: {len(dataset)}")
        print(f"Num generations per prompt: {num_generations}")
        print(f"Steps per generation: {steps_per_generation}")
        print(f"Temperature: {temperature}")
        print("-" * 50)
    
    # Validate dataset has the required column
    if prompt_column not in dataset.column_names:
        raise ValueError(f"Dataset must contain column '{prompt_column}'. Available columns: {dataset.column_names}")
    
    # Setup logging
    log_file_path, log_data = _setup_logging(log_file, log_format, model, verbose)
    
    # Update log metadata if JSON format
    if log_format == "json" and log_data:
        log_data["metadata"].update({
            "num_generations": num_generations,
            "temperature": temperature,
            "top_p": top_p,
            "max_completion_length": max_completion_length,
            "seed": seed
        })
    
    # Setup model and tokenizer
    _setup_model_and_tokenizer(model, tokenizer, seed, verbose)
    
    # Create sampler and dataloader
    dataloader = _create_sampler_and_dataloader(
        dataset, num_generations, per_device_batch_size, 
        steps_per_generation, seed, verbose
    )
    
    # Generation configuration
    generation_config = GenerationConfig(
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    reward_func_names = []
    processed_reward_funcs = []
    
    if reward_funcs is not None:
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        
        # Process each reward function
        for i, reward_func in enumerate(reward_funcs):
                # Custom callable reward function
                processed_reward_funcs.append(reward_func)
                reward_func_names.append(getattr(reward_func, '__name__', f'custom_reward_{i}'))
        
        if verbose:
            print(f"🎯 Loaded {len(processed_reward_funcs)} reward functions: {reward_func_names}")
    

    all_prompts = []
    all_completions = []
    completion_groups = {}
    reward_scores = {}  # Maps prompt to dict of reward function scores
    batch_stats = []
    
    if verbose:
        print("\n📋 Processing batches...")
    
    # Process batches
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            # Call _process_batch and collect outputs for aggregation
            (
                batch_maybe_conversational_prompts,
                batch_unique_prompts,
                batch_completions,
                batch_completion_groups,
                batch_reward_scores
            ) = _process_batch(
                batch,
                batch_idx,
                model,
                tokenizer,
                generation_config,
                processed_reward_funcs,
                reward_func_names,
                prompt_column,
                prompt_id_column,
                log_file_path,
                log_format,
                log_data,
                verbose
            )
            # Aggregate results across all batches
            all_prompts.extend(batch_maybe_conversational_prompts)
            all_completions.extend(batch_completions)
            for prompt in batch_completion_groups:
                if prompt not in completion_groups:
                    completion_groups[prompt] = []
                completion_groups[prompt].extend(batch_completion_groups[prompt])
            for prompt, func_scores in batch_reward_scores.items():
                if prompt not in reward_scores:
                    reward_scores[prompt] = {}
                for func_name, scores in func_scores.items():
                    if func_name not in reward_scores[prompt]:
                        reward_scores[prompt][func_name] = []
                    reward_scores[prompt][func_name].extend(scores)
            batch_stats.append({
                "batch_idx": batch_idx,
                "num_prompts": len(batch_maybe_conversational_prompts),
                "num_unique_prompts": len(batch_unique_prompts),
                "num_completions": len(batch_completions)
            })
    overall_stats, mean_reward_scores = _finalize_stats_and_logging(
        all_prompts,
        all_completions,
        completion_groups,
        reward_scores,
        reward_func_names,
        reward_funcs,
        batch_stats,
        log_file_path,
        log_format,
        log_data,
        verbose
    )
    
    if verbose:
        print(f"\n✅ Rollout demo complete!")
        if overall_stats:
            print(f"\n📈 Overall Statistics:")
            print(f"  Total prompts processed: {overall_stats['total_prompts_processed']}")
            print(f"  Unique prompts: {overall_stats['total_unique_prompts']}")
            print(f"  Total completions: {overall_stats['total_completions_generated']}")
            print(f"  Average completion length: {overall_stats['avg_completion_length']:.1f} words")
            
            if 'mean_rewards_per_function' in overall_stats:
                print(f"\n🏆 Reward Function Results:")
                for func_name, stats in overall_stats['mean_rewards_per_function'].items():
                    print(f"  {func_name}: mean={stats['mean']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}")
        
        print(f"\n🔑 Key Insights:")
        print(f"  • RepeatSampler created {num_generations} copies of each prompt")
        print(f"  • Single model.generate() call processed all duplicates")
        print(f"  • Sampling diversity from temperature={temperature}, top_p={top_p}")
        if reward_funcs:
            print(f"  • Computed rewards using {len(reward_func_names)} reward functions")
    
    return {
        'prompts': all_prompts,
        'completions': all_completions,
        'completion_groups': completion_groups,
        'reward_scores': mean_reward_scores,  # Mean rewards per unique prompt
        'stats': overall_stats,
        'batch_stats': batch_stats,
        'log_file_path': log_file_path
    }


def _calculate_rewards(
    prompt: str,
    completions: List[str],
    reward_funcs: List[RewardFunc],
    reward_func_names: List[str],
    inputs: Optional[List[Dict[str, Any]]] = None,
    completion_ids_list: Optional[List[torch.Tensor]] = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Calculate rewards for completions using the provided reward functions.
    Inspired by GRPOTrainer's _calculate_rewards method.
    """
    
    rewards_per_func = {}
    
    reward_kwargs = {}
    if inputs is not None:
        # Extract all input columns except "prompt", "completion", and "completion_ids"
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
    
    for i, (reward_func, reward_func_name) in enumerate(
        zip(reward_funcs, reward_func_names)
    ):
        try:
            output_rewards = reward_func(
                prompts=[prompt] * len(completions),
                completions=completions,
                completion_ids=completion_ids_list,
                **reward_kwargs
            )
            
            # Handle None values
            valid_rewards = [r for r in output_rewards if r is not None]
                
            rewards_per_func[reward_func_name] = valid_rewards
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Error computing reward with {reward_func_name}: {e}")
            raise Exception(e)
    
    return rewards_per_func


def _log_completions_to_file(
    prompt: str, 
    completions: List[str], 
    log_file_path: str, 
    log_format: str, 
    batch_idx: int,
    log_data: Optional[Dict] = None,
    reward_scores: Optional[Dict[str, float]] = None
):
    """Helper function to log completions to file in the specified format."""
    
    if log_format == "jsonl":
        # Append each completion as a separate JSON line
        with open(log_file_path, 'a', encoding='utf-8') as f:
            for i, completion in enumerate(completions):
                entry = {
                    "batch_idx": batch_idx,
                    "prompt": prompt,
                    "completion_idx": i,
                    "completion": completion,
                    "timestamp": datetime.now().isoformat(),
                }
                if reward_scores:
                    entry["reward_scores"] = reward_scores
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    elif log_format == "json":
        # Add to the main log_data structure
        if log_data is not None:
            for i, completion in enumerate(completions):
                entry = {
                    "batch_idx": batch_idx,
                    "prompt": prompt,
                    "completion_idx": i,
                    "completion": completion,
                    "timestamp": datetime.now().isoformat(),
                }
                if reward_scores:
                    entry["reward_scores"] = reward_scores
                log_data["completions"].append(entry)
    
    elif log_format == "txt":
        # Append in human-readable text format
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"BATCH {batch_idx + 1}\n")
            f.write(f"PROMPT: {prompt}\n")
            if reward_scores:
                f.write(f"REWARDS: {reward_scores}\n")
            f.write("-" * 60 + "\n")
            for i, completion in enumerate(completions):
                f.write(f"COMPLETION {i + 1}: {completion}\n")
            f.write("=" * 80 + "\n\n")


def create_sample_dataset() -> Dataset:
    """Create a simple dataset for demonstration purposes."""
    prompts = [
        "What is the capital of France?",
        "Explain quantum physics in simple terms.",
        "Write a short poem about nature.",
        "How do neural networks work?",
        "What are the benefits of exercise?"
    ]
    
    return Dataset.from_dict({"prompt": prompts})


# Example usage
def main():
    
    # Load a small model for demo
    model_name = "microsoft/DialoGPT-small"
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create sample dataset
    dataset = create_sample_dataset()
    
    # Run the demo with reward functions
    results = rollout_eval(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        prompt_column="prompt",
        prompt_id_column="prompt",
        num_generations=4,
        per_device_batch_size=2,
        steps_per_generation=2,
        max_completion_length=30,
        temperature=0.8,
        max_batches=2,
        verbose=True,
        log_file="demo_completions.jsonl",  # Specify log file
        log_format="jsonl",  # Can be "jsonl", "json", or "txt"
        reward_funcs=[simple_length_reward],  # Add reward functions
    )
    
    # Access results
    print(f"\n📊 Summary:")
    print(f"Processed {results['stats']['total_unique_prompts']} unique prompts")
    print(f"Generated {results['stats']['total_completions_generated']} total completions")
    
    # Show reward scores for each prompt
    if results['reward_scores']:
        print(f"\n🏆 Reward Scores by Prompt:")
        for prompt, scores in results['reward_scores'].items():
            print(f"'{prompt[:50]}...': {scores}")
    
    # Example: Get all completions for first prompt
    first_prompt = list(results['completion_groups'].keys())[0]
    first_prompt_completions = results['completion_groups'][first_prompt]
    print(f"\nFirst prompt: '{first_prompt}'")
    print(f"Generated {len(first_prompt_completions)} completions for this prompt")
    
    if results['reward_scores'] and first_prompt in results['reward_scores']:
        print(f"Mean rewards: {results['reward_scores'][first_prompt]}")

def simple_length_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Simple reward function that rewards longer completions.
    Returns a reward proportional to the length of the completion in words.
    """
    rewards = []
    for completion in completions:
        word_count = len(completion.split())
        reward = 5/max(5, word_count) if word_count >= 5 else 5/(5+(5 - word_count))
        rewards.append(reward)
    return rewards


if __name__ == "__main__":
    main()