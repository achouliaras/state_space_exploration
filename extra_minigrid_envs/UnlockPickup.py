from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Goal
from typing import Any, Iterable, SupportsFloat, TypeVar
from gymnasium.core import ActType, ObsType
import numpy as np

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class UnlockPickupEnv(RoomGrid):
    """
    ## Description

    The agent has to pick up a box which is placed in another room, behind a
    locked door. This environment can be solved without relying on language.

    ## Mission Space

    "get to the green goal square"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
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

    1. The agent picks up the correct box.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-UnlockPickup-v0`

    """

    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 5
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 16 * room_size**2

        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )
        self.first_time_door_open = True
        self.first_time_key_pickup = True

    @staticmethod
    def _gen_mission():
        return f"get to the green goal square"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # # Add a box to the room on the right
        # obj, _ = self.add_object(1, 0, kind="box")
        
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        key, _ = self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        # Place a goal in the room on the right
        obj, _ = self.place_in_room(1, 0, Goal())

        self.key = key
        self.door = door
        self.goal = obj
        self.mission = f"get to the green goal square"

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
        # elif np.array_equal(self.front_pos, self.door.cur_pos) and self.carrying and self.carrying == self.key and action != self.actions.toggle:
        #     # If the agent is at the door position and not toggling it, it is a failure
        #     reward += -0.2
        # elif self.carrying and self.carrying == self.key and action == self.actions.drop and self.door.is_locked:
        #     # Penalty if drop the key while door is locked
        #     reward += -0.2
        obs, reward, terminated, truncated, info = super().step(action)
        info["true_reward"] = 0
        # reward += self.reward_model()

        if self.agent_pos == self.goal.cur_pos:
            reward += self._reward()
            terminated = True
            info["true_reward"] = self._reward()
        elif action == self.actions.pickup:
            if self.carrying and self.carrying == self.key:
                reward += 0.2
                if self.first_time_key_pickup:
                    self.first_time_key_pickup = False
        elif action == self.actions.toggle:
            if self.door.is_open and self.first_time_door_open:
                reward += 0.2
                self.first_time_door_open = False

        return obs, reward, terminated, truncated, info
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        # Reset the first time door open and key pickup flags
        self.first_time_key_pickup = True
        self.first_time_door_open = True
        return super().reset(seed=seed)
    
    def reward_model(self):
        """
        Compute the reward to be given
        """
        reward = self._penalty()

        if not self.carrying and not self.door.is_open:
            dist = manhattan_distance(self.agent_pos, self.key.cur_pos)
        elif self.carrying == self.key and not self.door.is_open:
            dist = manhattan_distance(self.agent_pos, self.door.cur_pos)
        elif self.door.is_open:
            dist = manhattan_distance(self.agent_pos, self.goal.cur_pos)
        else:
            print(self.carrying)
        reward += -5*self._penalty() * (1 - (dist/(1+dist)))
        return reward
    