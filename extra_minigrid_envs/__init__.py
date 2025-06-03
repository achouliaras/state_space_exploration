from gymnasium.envs.registration import register
from extra_minigrid_envs.BlockedOpenDoor import BlockedOpenDoorEnv
from extra_minigrid_envs.BlockedPickup import BlockedPickupEnv
from extra_minigrid_envs.BlockedUnlock import BlockedUnlockEnv
from extra_minigrid_envs.OpenDoor import OpenDoorEnv
from extra_minigrid_envs.PickupKey import PickupKeyEnv
from extra_minigrid_envs.Pickup import PickupEnv    

def register_all():
    """Register all MiniGrid environments."""
    register(
            id="MiniGrid-BlockedOpenDoor-v0",
            entry_point="extra_minigrid_envs.BlockedOpenDoor:BlockedOpenDoorEnv",
        )
    register(
            id="MiniGrid-BlockedPickup-v0",
            entry_point="extra_minigrid_envs.BlockedPickup:BlockedPickupEnv",
        )
    register(
            id="MiniGrid-BlockedUnlock-v0",
            entry_point="extra_minigrid_envs.BlockedUnlock:BlockedUnlockEnv",
        )
    register(
            id="MiniGrid-OpenDoor-v0",
            entry_point="extra_minigrid_envs.OpenDoor:OpenDoorEnv",
        )
    register(
            id="MiniGrid-Pickup-v0",
            entry_point="extra_minigrid_envs.Pickup:PickupEnv",
        )
    register(
            id="MiniGrid-PickupKey-v0",
            entry_point="extra_minigrid_envs.PickupKey:PickupKeyEnv",
        )
    register(
            id="MiniGrid-Unlock-v0",
            entry_point="extra_minigrid_envs.Unlock:UnlockEnv",
        )
    register(
            id="MiniGrid-UnlockPickup-v0",
            entry_point="extra_minigrid_envs.UnlockPickup:UnlockPickupEnv",
        )
