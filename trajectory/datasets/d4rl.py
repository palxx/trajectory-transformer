import os
import numpy as np

## avoid huggingface_hub trying (and failing) to create symlinks without
## Windows developer mode / admin privileges
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import gymnasium
import minari
import pdb

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

## map from the D4RL-style env family name to the gymnasium MuJoCo env id
GYM_ENV_IDS = {
    'halfcheetah': 'HalfCheetah-v5',
    'hopper': 'Hopper-v5',
    'walker2d': 'Walker2d-v5',
    'ant': 'Ant-v5',
}

## extra kwargs needed so the gymnasium observation layout matches the
## original D4RL-v2 (mujoco-py) observation layout
GYM_ENV_KWARGS = {}

## reference scores used by D4RL to compute `get_normalized_score`
## (REF_MIN_SCORE / REF_MAX_SCORE from d4rl/infos.py, '-v2' locomotion tasks)
REF_MIN_SCORE = {
    'halfcheetah': -280.178953,
    'hopper': -20.272305,
    'walker2d': 1.629008,
    'ant': -325.6,
}
REF_MAX_SCORE = {
    'halfcheetah': 12135.0,
    'hopper': 3234.3,
    'walker2d': 4592.3,
    'ant': 3879.7,
}

## map from the D4RL-style dataset "quality" to a Minari `mujoco/<env>/<id>` dataset id
MINARI_QUALITY_IDS = {
    'random': 'simple-v0',
    'medium': 'medium-v0',
    'expert': 'expert-v0',
}

def parse_dataset_name(name):
    '''
        'halfcheetah-medium-expert-v2' -> ('halfcheetah', 'medium-expert', 'v2')
    '''
    env_name, *quality_parts, version = name.split('-')
    quality = '-'.join(quality_parts)
    return env_name, quality, version

def load_environment(name):
    env_name, quality, version = parse_dataset_name(name)
    gym_id = GYM_ENV_IDS[env_name]
    kwargs = GYM_ENV_KWARGS.get(env_name, {})

    with suppress_output():
        wrapped_env = gymnasium.make(gym_id, **kwargs)

    max_episode_steps = wrapped_env.spec.max_episode_steps
    env = wrapped_env.unwrapped
    env.max_episode_steps = max_episode_steps
    env.name = name

    ref_min = REF_MIN_SCORE[env_name]
    ref_max = REF_MAX_SCORE[env_name]
    env.get_normalized_score = lambda score: (score - ref_min) / (ref_max - ref_min)

    return env

def _flatten_minari_dataset(minari_id):
    '''
        downloads (if necessary) and flattens a Minari dataset into the
        flat D4RL-style dict expected by `qlearning_dataset_with_timeouts`:
        observations, actions, rewards, terminals, timeouts (all length N)
    '''
    print(f'[ datasets/d4rl ] Loading Minari dataset: {minari_id}')
    dataset = minari.load_dataset(minari_id, download=True)

    observations, actions, rewards, terminals, timeouts = [], [], [], [], []
    for episode in dataset.iterate_episodes():
        ## episode.observations has length T+1; drop the trailing observation
        ## so all arrays share length T (next_observations are unused downstream)
        observations.append(np.asarray(episode.observations)[:-1])
        actions.append(np.asarray(episode.actions))
        rewards.append(np.asarray(episode.rewards))
        terminals.append(np.asarray(episode.terminations))
        timeouts.append(np.asarray(episode.truncations))

    return {
        'observations': np.concatenate(observations, axis=0),
        'actions': np.concatenate(actions, axis=0),
        'rewards': np.concatenate(rewards, axis=0),
        'terminals': np.concatenate(terminals, axis=0),
        'timeouts': np.concatenate(timeouts, axis=0),
    }

def _concat_datasets(datasets):
    keys = datasets[0].keys()
    return {key: np.concatenate([d[key] for d in datasets], axis=0) for key in keys}

def get_dataset(name):
    env_name, quality, version = parse_dataset_name(name)

    if quality == 'medium-expert':
        ## D4RL constructs `medium-expert` by mixing the `medium` and `expert` datasets
        medium = _flatten_minari_dataset(f'mujoco/{env_name}/medium-v0')
        expert = _flatten_minari_dataset(f'mujoco/{env_name}/expert-v0')
        return _concat_datasets([medium, expert])

    if quality == 'medium-replay':
        minari_id = f'D4RL/{env_name}/medium-replay-v0'
        try:
            return _flatten_minari_dataset(minari_id)
        except Exception as e:
            raise NotImplementedError(
                f"[ datasets/d4rl ] Could not load '{minari_id}' from Minari ({e}). "
                f"'medium-replay' datasets are not currently mirrored for all "
                f"environments in Minari; see README for details."
            )

    if quality not in MINARI_QUALITY_IDS:
        raise ValueError(f'[ datasets/d4rl ] Unrecognized dataset quality: {quality}')

    minari_id = f'mujoco/{env_name}/{MINARI_QUALITY_IDS[quality]}'
    return _flatten_minari_dataset(minari_id)

def qlearning_dataset_with_timeouts(dataset, terminate_on_end=False, **kwargs):
    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        realdone_bool = bool(dataset['terminals'][i])
        final_timestep = dataset['timeouts'][i]

        if i < N - 1:
            done_bool += dataset['timeouts'][i] #+1]

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        realdone_.append(realdone_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_)[:,None],
        'terminals': np.array(done_)[:,None],
        'realterminals': np.array(realdone_)[:,None],
    }
