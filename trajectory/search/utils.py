import numpy as np
import torch
import pdb

from ..utils.arrays import to_torch

VALUE_PLACEHOLDER = 1e6

def make_prefix(discretizer, context, obs, rtg, prefix_context=True, device='cuda:0'):
    '''
        rtg : float, desired return-to-go conditioning the upcoming transition
    '''
    observation_dim = obs.size

    ## rtg occupies column 0 of the discretizer's thresholds (transition layout
    ## is rtg, obs, action, reward, value)
    rtg_discrete = discretizer.discretize(np.array([rtg]), subslice=[0, 1])
    rtg_discrete = to_torch(rtg_discrete, dtype=torch.long, device=device)

    obs_discrete = discretizer.discretize(obs, subslice=[1, 1+observation_dim])
    obs_discrete = to_torch(obs_discrete, dtype=torch.long, device=device)

    if prefix_context:
        prefix = torch.cat(context + [rtg_discrete, obs_discrete], dim=-1)
    else:
        prefix = torch.cat([rtg_discrete, obs_discrete], dim=-1)

    return prefix

def extract_actions(x, observation_dim, action_dim, t=None):
    assert x.shape[1] == 1 + observation_dim + action_dim + 2
    actions = x[:, 1+observation_dim:1+observation_dim+action_dim]
    if t is not None:
        return actions[t]
    else:
        return actions

def update_context(context, discretizer, observation, action, reward, rtg, max_context_transitions, device='cuda:0'):
    '''
        context : list of transitions
            [ tensor( transition_dim ), ... ]

        rtg : float, the return-to-go that conditioned this transition's action
    '''
    ## use a placeholder for value because input values are masked out by model
    rew_val = np.array([reward, VALUE_PLACEHOLDER])
    transition = np.concatenate([[rtg], observation, action, rew_val])

    ## discretize transition and convert to torch tensor
    transition_discrete = discretizer.discretize(transition)
    transition_discrete = to_torch(transition_discrete, dtype=torch.long, device=device)

    ## add new transition to context
    context.append(transition_discrete)

    ## crop context if necessary
    context = context[-max_context_transitions:]

    return context