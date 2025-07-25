from __future__ import annotations

from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from typing import Any, Iterable, SupportsFloat, TypeVar
from gymnasium.core import ActType, ObsType
import numpy as np

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class PickupKeyEnv(RoomGrid):
    """
    ## Description

    The agent has to pickup the key. This environment can be solved without
    relying on language.

    ## Mission Space

    "pick up the key"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Unused                    |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Unlock-v0`

    """

    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 5
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 8 * room_size**2

        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "pick up the key"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)

        # Add a key to unlock the door
        key, _ = self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.key = key
        self.door = door
        self.min_dist_to_point = manhattan_distance(self.agent_pos, self.key.cur_pos)
        self.max_dist = self.min_dist_to_point
        self.mission = "pick up the key"

    def _penalty(self) -> float:
        """
        Compute the reward to be given upon success
        """
        return - 0.8 * (1 / self.max_steps)
    
    def step(self, action):
        # reward = self._penalty()
        # terminated = False
        # if np.array_equal(self.front_pos, self.key.cur_pos) and action != self.actions.pickup:
        #     # If the agent is at the key position and not picking it up, it is a failure
        #     reward += -0.2
        obs, reward, terminated, truncated, info = super().step(action)
        info["true_reward"] = 0
        # reward += self.reward_model()

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.key:
                reward = self._reward() 
                terminated = True
                info["true_reward"] = self._reward()
        
        return obs, reward, terminated, truncated, info
    
    def reward_model(self):
        """
        Compute the reward to be given
        """
        reward = self._penalty()

        dist = manhattan_distance(self.agent_pos, self.key.cur_pos)
        if dist < self.min_dist_to_point:
            self.min_dist_to_point = dist
            reward += -5*self._penalty() * (1 - (dist/(1+dist)))
        return reward
