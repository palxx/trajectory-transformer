import os
from typing import Optional
import numpy as np
from os.path import join

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.datasets.d4rl import REF_MAX_SCORE, parse_dataset_name
from trajectory.search import (
    beam_plan,
    make_prefix,
    extract_actions,
    update_context,
)

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.offline'
    n_episodes: int = 5
    ## when both are set, each episode samples its own target_return uniformly from
    ## [target_return_min, target_return_max] instead of using a single fixed value -
    ## gives RTG-conditioning coverage across a range of desired performance levels
    target_return_min: Optional[float] = None
    target_return_max: Optional[float] = None

#######################
######## setup ########
#######################

args = Parser().parse_args('plan')

#######################
####### models ########
#######################

gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.gpt_loadpath,
        epoch=args.gpt_epoch, device=args.device)
dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,
        'data_config.pkl')

env = datasets.load_environment(args.dataset)

discretizer = dataset.discretizer
discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

value_fn = lambda x: discretizer.value_fn(x, args.percentile)
preprocess_fn = datasets.get_preprocess_fn(env.name)

randomize_target_return = args.target_return_min is not None and args.target_return_max is not None
if randomize_target_return:
    print(f'[ generate ] sampling a random target_return per episode from '
          f'[{args.target_return_min:.2f}, {args.target_return_max:.2f}]', flush=True)
elif args.target_return is None:
    env_name, *_ = parse_dataset_name(args.dataset)
    args.target_return = REF_MAX_SCORE[env_name]
    print(f'[ generate ] target_return: {args.target_return:.2f}', flush=True)
else:
    print(f'[ generate ] target_return: {args.target_return:.2f}', flush=True)

#######################
###### main loop ######
#######################

## per-episode transitions, in the same flat format as trajectory/datasets/d4rl.py's
## get_dataset() output, so these can be concatenated straight into the offline dataset
all_observations, all_actions, all_rewards, all_terminals, all_timeouts, all_target_returns = [], [], [], [], [], []
episode_scores = []

for ep in range(args.n_episodes):

    observation, _ = env.reset()
    total_reward = 0
    episode_target_return = (
        np.random.uniform(args.target_return_min, args.target_return_max)
        if randomize_target_return else args.target_return
    )
    rtg = episode_target_return
    context = []

    ep_observations, ep_actions, ep_rewards, ep_terminals, ep_timeouts, ep_target_returns = [], [], [], [], [], []

    T = env.max_episode_steps
    for t in range(T):

        ## keep the raw (unpreprocessed) observation for saving - SequenceDataset applies
        ## `preprocess_fn` itself when this data is loaded, so saving it pre-applied here
        ## would double-apply it
        raw_observation = observation
        obs_for_model = preprocess_fn(observation)

        ## rtg going into this step's plan/conditioning - i.e. the return still to achieve
        ## from here to the end of the episode (target_return minus what's been earned so far)
        ep_target_returns.append(rtg)

        if t % args.plan_freq == 0:
            prefix = make_prefix(discretizer, context, obs_for_model, rtg, args.prefix_context, device=args.device)

            sequence = beam_plan(
                gpt, value_fn, prefix,
                args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
                discount, args.max_context_transitions, verbose=args.verbose,
                k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
                discretizer=discretizer, rtg=rtg,
            )
        else:
            sequence = sequence[1:]

        sequence_recon = discretizer.reconstruct(sequence)
        action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)

        next_observation, reward, terminated, truncated, _ = env.step(action)
        terminal = terminated or truncated

        ep_observations.append(np.asarray(raw_observation).copy())
        ep_actions.append(np.asarray(action).copy())
        ep_rewards.append(reward)
        ep_terminals.append(terminated)
        ep_timeouts.append(truncated)

        total_reward += reward
        score = env.get_normalized_score(total_reward)
        context = update_context(context, discretizer, obs_for_model, action, reward, rtg, args.max_context_transitions, device=args.device)
        rtg = rtg - reward

        print(
            f'[ generate ] episode {ep} / {args.n_episodes} | target_return: {episode_target_return:.2f} | '
            f't: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f}', flush=True,
        )

        if terminal:
            break
        observation = next_observation

    ## `env` is `wrapped_env.unwrapped` (see load_environment in d4rl.py), which strips
    ## gymnasium's TimeLimit wrapper - so `truncated` from env.step() is always False, no
    ## matter how long the episode ran. Without an explicit boundary marker on the last
    ## step, downstream segmenting (trajectory/datasets/sequence.py's segment()) can't
    ## tell where one episode ends and the next begins, and will fold every episode in
    ## this file into one giant "trajectory". Mark it here if the real env didn't.
    if not ep_terminals[-1]:
        ep_timeouts[-1] = True

    all_observations.append(np.stack(ep_observations, axis=0))
    all_actions.append(np.stack(ep_actions, axis=0))
    all_rewards.append(np.array(ep_rewards))
    all_terminals.append(np.array(ep_terminals))
    all_timeouts.append(np.array(ep_timeouts))
    all_target_returns.append(np.array(ep_target_returns))
    episode_scores.append(score)
    print(f'[ generate ] episode {ep} finished | target_return: {episode_target_return:.2f} | '
          f'score: {score:.4f} | steps: {t + 1}', flush=True)

generated = {
    'observations': np.concatenate(all_observations, axis=0),
    'actions': np.concatenate(all_actions, axis=0),
    'rewards': np.concatenate(all_rewards, axis=0),
    'terminals': np.concatenate(all_terminals, axis=0),
    'timeouts': np.concatenate(all_timeouts, axis=0),
    'target_returns': np.concatenate(all_target_returns, axis=0),
}

out_dir = join(args.logbase, args.dataset, 'generated', args.gpt_loadpath)
os.makedirs(out_dir, exist_ok=True)
out_path = join(out_dir, f'episodes_{args.suffix}.npz')
np.savez(out_path, **generated)

print(f'\n[ generate ] Saved {args.n_episodes} episodes '
      f'({generated["observations"].shape[0]} transitions) to {out_path}')
print(f'[ generate ] mean score across episodes: {np.mean(episode_scores):.4f}')
