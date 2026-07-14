import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .ein import EinLinear

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        ## mask previous value estimates (transition layout: rtg, obs, action, reward, value)
        joined_dim = config.observation_dim + config.action_dim + 3
        self.mask.squeeze()[:,joined_dim-1::joined_dim] = 0
        ## boolean keep-mask for scaled_dot_product_attention (True = attend)
        self.attn_pdrop = config.attn_pdrop
        self.register_buffer("keep_mask", self.mask.bool())
        ##
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        ## [ B x n_heads x T x head_dim ]
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # fused causal self-attention (flash / memory-efficient kernel)
        ## [ B x n_heads x T x head_size ]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self.keep_mask[:,:,:T,:T],
            dropout_p=self.attn_pdrop if self.training else 0.0,
        )
        ## [ B x T x embedding_dim ]
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem (+1 for stop token)
        self.tok_emb = nn.Embedding(config.vocab_size * config.transition_dim + 1, config.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        # self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head = EinLinear(config.transition_dim, config.n_embd, config.vocab_size + 1, bias=False)

        self.vocab_size = config.vocab_size
        self.stop_token = config.vocab_size * config.transition_dim
        self.block_size = config.block_size
        self.observation_dim = config.observation_dim

        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim
        self.rtg_weight = config.rtg_weight
        self.action_weight = config.action_weight
        self.reward_weight = config.reward_weight
        self.value_weight = config.value_weight

        self.embedding_dim = config.n_embd
        self.apply(self._init_weights)

        ## set by distribute() for multi-GPU pipeline-parallel training;
        ## None means "everything lives on whatever single device .to() put it on"
        self.block_devices = None
        self.use_checkpoint = getattr(config, 'grad_checkpoint', False)

    def distribute(self, devices):
        """
        Splits self.blocks evenly across `devices` (a list of device strings/torch.devices)
        and pins the input embeddings to devices[0] / output head to devices[-1].

        This is plain model parallelism within a single process (each transformer block
        runs on one GPU, activations are handed off across the device boundary) rather than
        NCCL-based sharding (FSDP/DDP), because NCCL is not available for native Windows
        processes. It trades throughput (no overlap between GPUs) for the ability to fit a
        model whose parameters don't fit on a single GPU.

        Parameters are also cast to bf16 here (not just autocast'd during forward): autocast
        only casts activations, leaving fp32 weights + fp32 grads resident, which is roughly
        2x the memory a plain bf16 model needs and was enough by itself to blow past 48GB
        per GPU on a ~13B-parameter model.
        """
        self.devices = [torch.device(d) for d in devices]
        n_blocks = len(self.blocks)
        n_dev = len(self.devices)
        self.block_devices = [self.devices[i * n_dev // n_blocks] for i in range(n_blocks)]

        self.tok_emb.to(self.devices[0])
        self.pos_emb.data = self.pos_emb.data.to(self.devices[0])
        self.drop.to(self.devices[0])
        for block, device in zip(self.blocks, self.block_devices):
            block.to(device)
        self.ln_f.to(self.devices[-1])
        self.head.to(self.devices[-1])
        self.to(dtype=torch.bfloat16)
        return self

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EinLinear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        ## prefer bitsandbytes' paged 8-bit AdamW: momentum/variance are stored in 1 byte
        ## each instead of 4 (torch.optim.AdamW), and "paged" means CUDA transparently
        ## spills optimizer state to pinned host RAM under GPU memory pressure instead of
        ## OOMing. This is what makes a multi-billion parameter model's optimizer state
        ## feasible on 48GB cards without hand-rolled CPU offloading.
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
            print('[ models/transformers ] Using bitsandbytes PagedAdamW8bit (8-bit, CPU-paged optimizer state)')
        except ImportError:
            optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
            print('[ models/transformers ] bitsandbytes not installed; falling back to torch.optim.AdamW '
                  '(4x more optimizer memory, no CPU paging) - `pip install bitsandbytes` to enable it')
        return optimizer

    def offset_tokens(self, idx):
        _, t = idx.shape
        n_states = int(np.ceil(t / self.transition_dim))
        offsets = torch.arange(self.transition_dim) * self.vocab_size
        offsets = offsets.repeat(n_states).to(idx.device)
        offset_idx = idx + offsets[:t]
        offset_idx[idx == self.vocab_size] = self.stop_token
        return offset_idx

    def pad_to_full_observation(self, x, verify=False):
        b, t, _ = x.shape
        n_pad = (self.transition_dim - t % self.transition_dim) % self.transition_dim
        padding = torch.zeros(b, n_pad, self.embedding_dim, device=x.device, dtype=x.dtype)
        ## [ B x T' x embedding_dim ]
        x_pad = torch.cat([x, padding], dim=1)
        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        x_pad = x_pad.view(-1, self.transition_dim, self.embedding_dim)
        if verify:
            self.verify(x, x_pad)
        return x_pad, n_pad

    def verify(self, x, x_pad):
        b, t, embedding_dim = x.shape
        n_states = int(np.ceil(t / self.transition_dim))
        inds = torch.arange(0, self.transition_dim).repeat(n_states)[:t]
        for i in range(self.transition_dim):
            x_ = x[:,inds == i]
            t_ = x_.shape[1]
            x_pad_ = x_pad[:,i].view(b, n_states, embedding_dim)[:,:t_]
            print(i, x_.shape, x_pad_.shape)
            try:
                assert (x_ == x_pad_).all()
            except:
                pdb.set_trace()

    def forward(self, idx, targets=None, mask=None):
        """
            idx : [ B x T ]
            values : [ B x 1 x 1 ]
        """
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.block_devices is not None:
            idx = idx.to(self.devices[0])

        offset_idx = self.offset_tokens(idx)
        ## [ B x T x embedding_dim ]
        # forward the GPT model
        token_embeddings = self.tok_emb(offset_idx) # each index maps to a (learnable) vector
        ## [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        ## [ B x T x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings)

        if self.block_devices is None:
            x = self.blocks(x)
        else:
            ## pipeline-parallel forward: hand activations across the GPU boundary
            ## wherever a block lives on a different device than the previous one
            for block, device in zip(self.blocks, self.block_devices):
                if x.device != device:
                    x = x.to(device)
                if self.use_checkpoint and self.training:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        ## [ B x T x embedding_dim ]
        x = self.ln_f(x)

        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        x_pad, n_pad = self.pad_to_full_observation(x)
        ## [ (B * T' / transition_dim) x transition_dim x (vocab_size + 1) ]
        logits = self.head(x_pad)
        ## [ B x T' x (vocab_size + 1) ]
        logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)
        ## [ B x T x (vocab_size + 1) ]
        logits = logits[:,:t]

        # if we are given some desired targets also calculate the loss
        if targets is not None:
            targets = targets.to(logits.device)
            mask = mask.to(logits.device)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), reduction='none')
            if self.rtg_weight != 1 or self.action_weight != 1 or self.reward_weight != 1 or self.value_weight != 1:
                #### make weights
                n_states = int(np.ceil(t / self.transition_dim))
                weights = torch.cat([
                    torch.ones(1, device=logits.device) * self.rtg_weight,
                    torch.ones(self.observation_dim, device=logits.device),
                    torch.ones(self.action_dim, device=logits.device) * self.action_weight,
                    torch.ones(1, device=logits.device) * self.reward_weight,
                    torch.ones(1, device=logits.device) * self.value_weight,
                ])
                ## [ t + 1]
                weights = weights.repeat(n_states)
                ## [ b x t ]
                weights = weights[1:].repeat(b, 1)
                ####
                loss = loss * weights.view(-1)
            loss = (loss * mask.view(-1)).mean()
        else:
            loss = None

        return logits, loss

class ConditionalGPT(GPT):

    def __init__(self, config):
        ## increase block size by `observation_dim` because we are prepending a goal observation
        ## to the sequence
        config.block_size += config.observation_dim
        super().__init__(config)
        self.goal_emb = nn.Embedding(config.vocab_size * config.observation_dim, config.n_embd)

    def get_block_size(self):
        return self.block_size - self.observation_dim

    def forward(self, idx, goal, targets=None, mask=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        #### goal
        offset_goal = self.offset_tokens(goal)
        goal_embeddings = self.goal_emb(offset_goal)
        #### /goal

        offset_idx = self.offset_tokens(idx)
        ## [ B x T x embedding_dim ]
        # forward the GPT model
        token_embeddings = self.tok_emb(offset_idx) # each index maps to a (learnable) vector
        ## [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        ## [ B x T x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings)

        #### goal
        ## [ B + (obs_dim + T) x embedding_dim ]
        gx = torch.cat([goal_embeddings, x], dim=1)
        gx = self.blocks(gx)
        x = gx[:, self.observation_dim:]
        #### /goal

        ## [ B x T x embedding_dim ]
        x = self.ln_f(x)

        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        x_pad, n_pad = self.pad_to_full_observation(x)
        ## [ (B * T' / transition_dim) x transition_dim x (vocab_size + 1) ]
        logits = self.head(x_pad)
        ## [ B x T' x (vocab_size + 1) ]
        logits = logits.reshape(b, t + n_pad, self.vocab_size + 1)
        ## [ B x T x (vocab_size + 1) ]
        logits = logits[:,:t]

        # if we are given some desired targets also calculate the loss
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), reduction='none')
            if self.rtg_weight != 1 or self.action_weight != 1 or self.reward_weight != 1 or self.value_weight != 1:
                #### make weights
                n_states = int(np.ceil(t / self.transition_dim))
                weights = torch.cat([
                    torch.ones(1, device=idx.device) * self.rtg_weight,
                    torch.ones(self.observation_dim, device=idx.device),
                    torch.ones(self.action_dim, device=idx.device) * self.action_weight,
                    torch.ones(1, device=idx.device) * self.reward_weight,
                    torch.ones(1, device=idx.device) * self.value_weight,
                ])
                ## [ t + 1]
                weights = weights.repeat(n_states)
                ## [ b x t ]
                weights = weights[1:].repeat(b, 1)
                ####
                loss = loss * weights.view(-1)
            loss = (loss * mask.view(-1)).mean()
        else:
            loss = None

        return logits, loss
