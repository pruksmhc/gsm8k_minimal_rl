

## Requirements

```bash
pip install torch transformers datasets trl flash-attention
```
python generate_rollout.py for the simple dataset. 

TODO: 
-Make sure the GSMK training still works. 
-Make the rollouts work with GSMK with the chat template. 

The steps is regardless of batch size - more steps per update with larger batch. 

Wandb run with per gpu batch size of 64 (1 gpu): https://wandb.ai/yada-pruksachatkun/huggingface/runs/qwx7eqko?nw=nwuseryadapruksachatkun 

Wandb run with per gpu batch size of 8 - it converges slower: https://wandb.ai/yada-pruksachatkun/huggingface/runs/0bnoneu5?nw=nwuseryadapruksachatkun . Less diversity in batch. 