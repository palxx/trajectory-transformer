import os
from typing import Optional
import numpy as np
import torch
import pdb

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.models.transformers import GPT

## allow TF32 matmuls/convs on Ampere+ GPUs (RTX 6000 Ada) for faster fp32 ops
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.offline'
    ## path to a .npz saved by scripts/generate.py; when set, those self-generated
    ## rollout transitions are mixed into the offline dataset before training
    generated_data_path: Optional[str] = None

#######################
######## setup ########
#######################

args = Parser().parse_args('train')

#######################
####### dataset #######
#######################

env = datasets.load_environment(args.dataset)

sequence_length = args.subsampled_sequence_length * args.step

dataset_config = utils.Config(
    datasets.DiscretizedDataset,
    savepath=(args.savepath, 'data_config.pkl'),
    env=args.dataset,
    N=args.N,
    penalty=args.termination_penalty,
    sequence_length=sequence_length,
    step=args.step,
    discount=args.discount,
    discretizer=args.discretizer,
    generated_data_path=args.generated_data_path,
)

dataset = dataset_config()
obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
transition_dim = dataset.joined_dim

#######################
######## model ########
#######################

block_size = args.subsampled_sequence_length * transition_dim - 1
print(
    f'Dataset size: {len(dataset)} | '
    f'Joined dim: {transition_dim} '
    f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}'
)

model_config = utils.Config(
    GPT,
    savepath=(args.savepath, 'model_config.pkl'),
    ## discretization
    vocab_size=args.N, block_size=block_size,
    ## architecture
    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd*args.n_head,
    ## dimensions
    observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
    ## loss weighting
    rtg_weight=args.rtg_weight,
    action_weight=args.action_weight, reward_weight=args.reward_weight, value_weight=args.value_weight,
    ## dropout probabilities
    embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop,
    ## recompute activations during backward instead of storing them, to trade
    ## compute time for GPU memory (helps most when the model doesn't fit otherwise)
    grad_checkpoint=True,
)

model = model_config()

## NCCL (needed for torch.distributed / FSDP) isn't available for native Windows
## processes, so with >1 GPU we fall back to manual pipeline-parallelism: split the
## transformer blocks across the visible GPUs within this single process instead of
## sharding via a distributed backend. Slower than data parallel, but doesn't require
## the whole model to fit on one card.
##
## Splitting only helps when the model doesn't comfortably fit on one GPU - for small
## models it's pure overhead (pipeline bubble + cross-device transfer, zero memory
## benefit), so only do it if a rough bf16 weights+grad estimate would eat more than
## 70% of a single card.
n_params = sum(p.numel() for p in model.parameters())
bf16_train_bytes = n_params * 4  # ~2 bytes bf16 weights + ~2 bytes bf16 grads
n_gpus = torch.cuda.device_count()
single_gpu_capacity = torch.cuda.get_device_properties(0).total_memory * 0.7 if torch.cuda.is_available() else 0
needs_split = bf16_train_bytes > single_gpu_capacity
print(f'[ train ] Model has {n_params/1e6:.1f}M params (~{bf16_train_bytes/1e9:.2f}GB bf16 weights+grads)')

if n_gpus > 1 and 'cuda' in args.device and needs_split:
    devices = [f'cuda:{i}' for i in range(n_gpus)]
    print(f'[ train ] {n_gpus} GPUs visible; splitting model across {devices}')
    model.distribute(devices)
    train_device = devices[0]
    ## split each batch into microbatches and pipeline them across the GPU split
    ## (see Trainer.train) so GPU0 isn't idle while GPU1 finishes the previous
    ## microbatch, instead of sending the whole batch through as one lockstep unit.
    num_microbatches = min(4, args.batch_size)
else:
    model.to(args.device)
    train_device = args.device
    num_microbatches = 1

#######################
####### trainer #######
#######################

warmup_tokens = len(dataset) * block_size ## number of tokens seen per epoch
final_tokens = 20 * warmup_tokens

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    # optimization parameters
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=0.1, # only applied on matmul weights
    # learning rate decay: linear warmup followed by cosine decay to 10% of original
    lr_decay=args.lr_decay,
    warmup_tokens=warmup_tokens,
    final_tokens=final_tokens,
    ## dataloader
    num_workers=0,
    device=train_device,
    num_microbatches=num_microbatches,
)

trainer = trainer_config()

#######################
###### main loop ######
#######################

## scale number of epochs to keep number of updates constant
n_epochs = int(1e6 / len(dataset) * args.n_epochs_ref)
save_freq = max(int(n_epochs // args.n_saves), 1)

for epoch in range(n_epochs):
    print(f'\nEpoch: {epoch} / {n_epochs} | {args.dataset} | {args.exp_name}', flush=True)

    epoch_loss = trainer.train(model, dataset)

    ## log + (re)plot the training curve, so progress is visible while training runs
    utils.log_training_loss(args.savepath, epoch, epoch_loss)
    utils.plot_training_curve(args.savepath)

    ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
    save_epoch = (epoch + 1) // save_freq * save_freq
    statepath = os.path.join(args.savepath, f'state_{save_epoch}.pt')
    print(f'Saving model to {statepath}', flush=True)

    ## save state to disk
    state = model.state_dict()
    torch.save(state, statepath)
