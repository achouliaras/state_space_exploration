from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball
from typing import Any, Iterable, SupportsFloat, TypeVar
from gymnasium.core import ActType, ObsType

class BlockedUnlockEnv(RoomGrid):
    """

    ## Description

    The agent has to pick up a box which is placed in another room, behind a
    locked door. The door is also blocked by a ball which the agent has to move
    before it can open the door. Hence, the agent has to learn to move the
    ball, unlock and open the door. This environment can be solved without relying on language.

    ## Mission Space

    "open the door"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

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

    - `MiniGrid-BlockedUnlockPickup-v0`

    """

    def __init__(self, max_steps: int | None = None, **kwargs):
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, ["box", "key"]],
        )

        room_size = 5
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
        self.first_time_key_pickup = True

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"open the door"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # # Add a box to the room on the right
        # obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by an unlocked door
        door, pos = self.add_door(0, 0, 0, locked=False)
        # Block the door with a ball
        color = self._rand_color()
        self.grid.set(pos[0] - 1, pos[1], Ball(color))
        
        # Add a key to unlock the door
        key, _ = self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.key = key
        self.door = door
        self.mission = f"open the door"

    def _penalty(self) -> float:
        """
        Compute the reward to be given upon success
        """
        return - 0.8 * (1 / self.max_steps)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        info["true_reward"] = 0
        reward = self._penalty()

        if action == self.actions.toggle:
            if self.door.is_open:
                reward += 1
                terminated = True
                info["true_reward"] = self._reward()
        elif action == self.actions.pickup:
            if self.first_time_key_pickup and self.carrying and self.carrying == self.key:
                reward += 0.2
                self.first_time_key_pickup = False

        return obs, reward, terminated, truncated, info
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        # Reset the first time key pickup flag
        self.first_time_key_pickup = True
        return super().reset(seed=seed)