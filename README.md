
Bare-bones implementation of a RL experiment on GSMK. 

## Installation

```bash
pip install torch transformers datasets trl flash-attn
```

## Commands
```bash
python generate_rollout.py to test rollouts on a simple sample dataset. 
python train
```
Total steps was around 234 with larger batch size, so checkpoint 200 is near end of 1 epoch.  

Wandb run with per device batch size of 64 and 16 generation: https://wandb.ai/yada-pruksachatkun/huggingface/runs/qwx7eqko?nw=nwuseryadapruksachatkun 

Wandb run with per device batch size of 8 and 16 generation (slower convergence on a step-equivalent basis): https://wandb.ai/yada-pruksachatkun/huggingface/runs/0bnoneu5?nw=nwuseryadapruksachatkun, probably due to more variance per step due to less examples per batch. 

Notes: 
* For RepeatSampler, it will repeat an item contiguously within a batch. Effective generation batch size is larger than per_device_batch_size * num_devices.  

TODO: 
* Look through the train set rollout where the model at checkpoint 200 is still getting things wrong, and adjust reward functions/do more sft to improve.
