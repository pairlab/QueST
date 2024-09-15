"""
The purpose of this file is to fix the silly gymnasium frame stack wrapper behavior I complain about here
https://github.com/Farama-Foundation/Gymnasium/issues/1085
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Final, SupportsFloat

import numpy as np

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
from gymnasium.wrappers.utils import create_zero_array


class FrameStackObservationFixed(
    gym.Wrapper[WrapperObsType, ActType, ObsType, ActType],
    gym.utils.RecordConstructorArgs,
):
    """Stacks the observations from the last ``N`` time steps in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Users have options for the padded observation used:

     * "reset" (default) - The reset value is repeated
     * "zero" - A "zero"-like instance of the observation space
     * custom - An instance of the observation space

    No vector version of the wrapper exists.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FrameStackObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env = FrameStackObservation(env, stack_size=4)
        >>> env.observation_space
        Box(0, 255, (4, 96, 96, 3), uint8)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)

    Example with different padding observations:
        >>> env = gym.make("CartPole-v1")
        >>> env.reset(seed=123)
        (array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), {})
        >>> stacked_env = FrameStackObservation(env, 3)   # the default is padding_type="reset"
        >>> stacked_env.reset(seed=123)
        (array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
              dtype=float32), {})


        >>> stacked_env = FrameStackObservation(env, 3, padding_type="zero")
        >>> stacked_env.reset(seed=123)
        (array([[ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
              dtype=float32), {})
        >>> stacked_env = FrameStackObservation(env, 3, padding_type=np.array([1, -1, 0, 2], dtype=np.float32))
        >>> stacked_env.reset(seed=123)
        (array([[ 1.        , -1.        ,  0.        ,  2.        ],
               [ 1.        , -1.        ,  0.        ,  2.        ],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
              dtype=float32), {})

    Change logs:
     * v0.15.0 - Initially add as ``FrameStack`` with support for lz4
     * v1.0.0 - Rename to ``FrameStackObservation`` and remove lz4 and ``LazyFrame`` support
                along with adding the ``padding_type`` parameter

    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        stack_size: int,
        *,
        padding_type: str | ObsType = "reset",
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env: The environment to apply the wrapper
            stack_size: The number of frames to stack.
            padding_type: The padding type to use when stacking the observations, options: "reset", "zero", custom obs
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, stack_size=stack_size, padding_type=padding_type
        )
        gym.Wrapper.__init__(self, env)

        if not np.issubdtype(type(stack_size), np.integer):
            raise TypeError(
                f"The stack_size is expected to be an integer, actual type: {type(stack_size)}"
            )
        # if not 1 < stack_size:
        #     raise ValueError(
        #         f"The stack_size needs to be greater than one, actual value: {stack_size}"
        #     )
        if isinstance(padding_type, str) and (
            padding_type == "reset" or padding_type == "zero"
        ):
            self.padding_value: ObsType = create_zero_array(env.observation_space)
        elif padding_type in env.observation_space:
            self.padding_value = padding_type
            padding_type = "_custom"
        else:
            if isinstance(padding_type, str):
                raise ValueError(  # we are guessing that the user just entered the "reset" or "zero" wrong
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r}"
                )
            else:
                raise ValueError(
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r} not an instance of env observation ({env.observation_space})"
                )

        self.observation_space = batch_space(env.observation_space, n=stack_size)
        self.stack_size: Final[int] = stack_size
        self.padding_type: Final[str] = padding_type

        self.obs_queue = deque(
            [self.padding_value for _ in range(self.stack_size)], maxlen=self.stack_size
        )
        self.stacked_obs = create_empty_array(env.observation_space, n=self.stack_size)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and info from the environment
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_queue.append(obs)

        updated_obs = deepcopy(
            concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
        )
        return updated_obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, 
        options: dict[str, Any] | None = None,
        **kwargs
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Reset the environment, returning the stacked observation and info.

        Args:
            seed: The environment seed
            options: The reset options

        Returns:
            The stacked observations and info
        """
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)

        if self.padding_type == "reset":
            self.padding_value = obs
        for _ in range(self.stack_size - 1):
            self.obs_queue.append(self.padding_value)
        self.obs_queue.append(obs)

        updated_obs = deepcopy(
            concatenate(self.env.observation_space, self.obs_queue, self.stacked_obs)
        )
        return updated_obs, info
