
Bare-bones implementation of a RL experiment on GSMK. This can be run on a 40GB A40. 

Qualitatively, from the rollouts there is a major difference in formatting and accuracy between checkpoint 0 and 200, reflecting the train loss and rewards.

## Installation

```bash
pip install transformers datasets trl flash-attn
```

## Commands
```bash
python generate_rollout.py # To test rollouts on a simple sample dataset. 
python train_gsmk.py --mode train  # To train a model on GSMK. 
```

Wandb run with per device batch size of 64 and 16 generation with Qwen2.5-1.5B-Instruct: https://wandb.ai/yada-pruksachatkun/huggingface/runs/qwx7eqko?nw=nwuseryadapruksachatkun. Total steps was around 234 with larger batch size, so checkpoint 200 is near end of 1 epoch.  


Wandb run with per device batch size of 8 and 16 generation (slower convergence on a step-equivalent basis) with Qwen2.5-1.5B-Instruct: https://wandb.ai/yada-pruksachatkun/huggingface/runs/0bnoneu5?nw=nwuseryadapruksachatkun, probably due to more variance per step due to less examples per batch. 

Notes: 
* For RepeatSampler, it will repeat an item contiguously within a batch. Effective generation batch size is larger than per_device_batch_size * num_devices.  

At checkpoint=200 steps, 62.65% of the examples have at least 4/8 generations correct, with temperature=0.8. This is in the ballpark of numbers reported in the Qwen2.5 paper. 
TODO: 
* Use a different sampler than RepeatSampler for rollouts. For GSM8K evals, with RepeatSampler, batch size must divide 1319 (test set size), otherwise RepeatSampler will drop the last remainder. 
* Look through the train set rollout where the model at checkpoint 200 is still getting things wrong, and adjust reward functions/do more sft to improve.
* Add proper evals for GSMK
