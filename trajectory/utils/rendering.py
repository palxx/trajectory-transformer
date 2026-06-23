import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import mujoco
import pdb

from .arrays import to_np
from .video import save_video, save_videos
from ..datasets import load_environment, get_preprocess_fn

def make_renderer(args):
    render_str = getattr(args, 'renderer')
    render_class = getattr(sys.modules[__name__], render_str)
    ## get dimensions in case the observations are preprocessed
    env = load_environment(args.dataset)
    preprocess_fn = get_preprocess_fn(args.dataset)
    observation, _ = env.reset()
    observation = preprocess_fn(observation)
    return render_class(args.dataset, observation_dim=observation.size)

def split(sequence, observation_dim, action_dim):
    assert sequence.shape[1] == observation_dim + action_dim + 2
    observations = sequence[:, :observation_dim]
    actions = sequence[:, observation_dim:observation_dim+action_dim]
    rewards = sequence[:, -2]
    values = sequence[:, -1]
    return observations, actions, rewards, values

def set_state(env, state):
    qpos_dim = env.data.qpos.size
    qvel_dim = env.data.qvel.size
    qstate_dim = qpos_dim + qvel_dim

    if 'ant' in env.name:
        ypos = np.zeros(1)
        state = np.concatenate([ypos, state])

    if state.size == qpos_dim - 1 or state.size == qstate_dim - 1:
        xpos = np.zeros(1)
        state = np.concatenate([xpos, state])

    if state.size == qpos_dim:
        qvel = np.zeros(qvel_dim)
        state = np.concatenate([state, qvel])

    if 'ant' in env.name and state.size > qpos_dim + qvel_dim:
        xpos = np.zeros(1)
        state = np.concatenate([xpos, state])[:qstate_dim]

    assert state.size == qpos_dim + qvel_dim

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollout_from_state(env, state, actions):
    qpos_dim = env.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, terminated, truncated, _ = env.step(act)
        observations.append(obs)
        if terminated or truncated:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)

class DebugRenderer:

    def __init__(self, *args, **kwargs):
        pass

    def render(self, *args, **kwargs):
        return np.zeros((10, 10, 3))

    def render_plan(self, *args, **kwargs):
        pass

    def render_rollout(self, *args, **kwargs):
        pass

class Renderer:

    def __init__(self, env, observation_dim=None, action_dim=None, dim=256):
        if type(env) is str:
            self.env = load_environment(env)
        else:
            self.env = env

        self.observation_dim = observation_dim or np.prod(self.env.observation_space.shape)
        self.action_dim = action_dim or np.prod(self.env.action_space.shape)

        self.dim = (dim, dim) if type(dim) is int else dim
        self.viewer = mujoco.Renderer(self.env.model, height=self.dim[1], width=self.dim[0])
        self.camera = mujoco.MjvCamera()

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

    def render(self, observation, dim=None, render_kwargs=None):
        observation = to_np(observation)

        if render_kwargs is None:
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [0, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.camera.lookat[:] = val[:]
            else:
                setattr(self.camera, key, val)

        set_state(self.env, observation)

        dim = (dim, dim) if type(dim) is int else (dim or self.dim)
        if dim != self.dim:
            self.viewer.close()
            self.dim = dim
            self.viewer = mujoco.Renderer(self.env.model, height=self.dim[1], width=self.dim[0])

        self.viewer.update_scene(self.env.data, camera=self.camera)
        data = self.viewer.render()
        return data

    def renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def render_plan(self, savepath, sequence, state, fps=30):
        '''
            state : np.array[ observation_dim ]
            sequence : np.array[ horizon x transition_dim ]
                as usual, sequence is ordered as [ s_t, a_t, r_t, V_t, ... ]
        '''

        if len(sequence) == 1:
            return

        sequence = to_np(sequence)

        ## compare to ground truth rollout using actions from sequence
        actions = sequence[:-1, self.observation_dim : self.observation_dim + self.action_dim]
        rollout_states = rollout_from_state(self.env, state, actions)

        videos = [
            self.renders(sequence[:, :self.observation_dim]),
            self.renders(rollout_states),
        ]

        save_videos(savepath, *videos, fps=fps)

    def render_rollout(self, savepath, states, **video_kwargs):
        images = self(states)
        save_video(savepath, images, **video_kwargs)

#--------------------------------- legacy renderers ---------------------------------#
#
# `KitchenRenderer`, `AntMazeRenderer`, and `Maze2dRenderer` were written against the
# original `gym==0.18` + `d4rl` + `mujoco-py` stack for the kitchen / antmaze / maze2d
# task families. Those datasets are not part of this migration (see README), and `gym`
# / `mujoco_py` are no longer dependencies of this project, so these renderers are kept
# only for reference and will raise on use.
#
#--------------------------------------------------------------------------------------#

ANTMAZE_BOUNDS = {
    'antmaze-umaze-v0': (-3, 11),
    'antmaze-medium-play-v0': (-3, 23),
    'antmaze-medium-diverse-v0': (-3, 23),
    'antmaze-large-play-v0': (-3, 39),
    'antmaze-large-diverse-v0': (-3, 39),
}

class _UnsupportedLegacyRenderer:

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f'[ utils/rendering ] {type(self).__name__} relies on the legacy '
            f'gym==0.18 / d4rl / mujoco-py stack and was not migrated. '
            f'See README for details.'
        )

class KitchenRenderer(_UnsupportedLegacyRenderer):
    pass

class AntMazeRenderer(_UnsupportedLegacyRenderer):
    pass

class Maze2dRenderer(_UnsupportedLegacyRenderer):
    pass

#--------------------------------- planning callbacks ---------------------------------#
