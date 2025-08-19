

## Requirements

```bash
pip install torch transformers datasets trl flash-attention
```

python generate_rollout.py to test rollouts on a simple sample dataset. 

The steps is regardless of batch size - more steps per update with larger batch. 
Check out the rollouts to see the improvements. Total steps was around 234 with larger batch size, so checkpoint 200 is near end of 1 epoch.  

Wandb run with per gpu batch size of 64 (1 gpu): https://wandb.ai/yada-pruksachatkun/huggingface/runs/qwx7eqko?nw=nwuseryadapruksachatkun 

Wandb run with per gpu batch size of 8 - it converges slower: https://wandb.ai/yada-pruksachatkun/huggingface/runs/0bnoneu5?nw=nwuseryadapruksachatkun . Less diversity in batch. 
