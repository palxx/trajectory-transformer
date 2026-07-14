import math
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import pdb

from .timer import Timer

def to(xs, device):
    return [x.to(device) for x in xs]

class Trainer:

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.n_epochs = 0
        self.n_tokens = 0 # counter used for learning rate decay
        self.optimizer = None

    def get_optimizer(self, model):
        if self.optimizer is None:
            print(f'[ utils/training ] Making optimizer at epoch {self.n_epochs}')
            self.optimizer = model.configure_optimizers(self.config)
        return self.optimizer

    def train(self, model, dataset, n_epochs=1, log_freq=100):

        config = self.config
        optimizer = self.get_optimizer(model)
        model.train(True)
        vocab_size = dataset.N
        num_microbatches = max(1, getattr(config, 'num_microbatches', 1))

        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

        epoch_loss = None
        for _ in range(n_epochs):

            losses = []
            timer = Timer()
            for it, batch in enumerate(loader):

                idx, targets, mask = to(batch, self.device)
                model.zero_grad()

                if num_microbatches > 1:
                    idx_chunks = idx.chunk(num_microbatches, dim=0)
                    target_chunks = targets.chunk(num_microbatches, dim=0)
                    mask_chunks = mask.chunk(num_microbatches, dim=0)
                    n_chunks = len(idx_chunks)

                    ## Pipeline-parallel overlap: dispatch every microbatch's forward pass
                    ## before dispatching any backward. A CUDA stream executes a device's
                    ## queued kernels strictly in enqueue order, so if forward+backward were
                    ## interleaved per microbatch, GPU0 would stall on microbatch k's
                    ## backward (which waits on GPU1) before it could start microbatch k+1's
                    ## forward - killing overlap. Queuing all forwards first means GPU0's
                    ## forward-only kernels for k+1..n have no such dependency and run back
                    ## to back, while GPU1 works through earlier microbatches concurrently;
                    ## the same reasoning applies in reverse once all backwards are queued.
                    micro_losses = []
                    for mi, mt, mm in zip(idx_chunks, target_chunks, mask_chunks):
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled='cuda' in self.device):
                            _, loss_mb = model(mi, mt, mm)
                        micro_losses.append(loss_mb / n_chunks)

                    for loss_mb in micro_losses:
                        loss_mb.backward()

                    loss = sum(l.detach() for l in micro_losses)
                else:
                    with torch.set_grad_enabled(True):
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled='cuda' in self.device):
                            logits, loss = model(idx, targets, mask)
                    loss.backward()

                losses.append(loss.item())

                # backprop and update the parameters
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                # decay the learning rate based on our progress
                if config.lr_decay:
                    y = targets
                    self.n_tokens += (y != vocab_size).sum() # number of tokens processed this step
                    if self.n_tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.n_tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.n_tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                # report progress
                if it % log_freq == 0:
                    print(
                        f'[ utils/training ] epoch {self.n_epochs} [ {it:4d} / {len(loader):4d} ] ',
                        f'train loss {loss.item():.5f} | lr {lr:.3e} | lr_mult: {lr_mult:.4f} | '
                        f't: {timer():.2f}', flush=True)

            self.n_epochs += 1
            epoch_loss = float(np.mean(losses))

        return epoch_loss
