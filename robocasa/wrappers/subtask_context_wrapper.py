from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import gymnasium as gym

import robocasa.utils.object_utils as OU


@dataclass
class SubtaskDef:
    description: str
    is_complete: Callable  # (kitchen_env) -> bool: True when this subtask is DONE


# ---------------------------------------------------------------------------
# Subtask registry for all 16 composite-seen tasks.
# Each task maps to an ordered list of SubtaskDefs.
# Current subtask = first entry whose is_complete() returns False.
# The final entry always has is_complete = lambda env: True (terminal).
# ---------------------------------------------------------------------------
SUBTASK_REGISTRY: dict[str, List[SubtaskDef]] = {
    "DeliverStraw": [
        SubtaskDef(
            "Pick up the straw from the drawer",
            lambda env: OU.check_obj_in_receptacle(env, "straw", "glass_cup", th=0.5),
        ),
        SubtaskDef(
            "Place the straw inside the glass cup on the dining counter",
            lambda env: True,
        ),
    ],
    "GetToastedBread": [
        SubtaskDef(
            "Push the toaster lever down to start toasting the bread",
            lambda env: env.toaster_on,
        ),
        SubtaskDef(
            "Wait for the lever to pop up, then remove the bread from the toaster",
            lambda env: not any(
                env.toaster.check_slot_contact(env, "obj", slot_pair=sp)
                for sp in range(len(env.toaster.get_state(env).keys()))
            ),
        ),
        SubtaskDef(
            "Place the toasted bread on the plate on the dining counter",
            lambda env: True,
        ),
    ],
    "KettleBoiling": [
        SubtaskDef(
            "Pick up the kettle from the counter and place it on a stove burner",
            lambda env: OU.check_obj_fixture_contact(env, "obj", env.stove),
        ),
        SubtaskDef(
            "Turn the burner on to start boiling",
            lambda env: True,
        ),
    ],
    "LoadDishwasher": [
        SubtaskDef(
            "Pick up the dishes from the counter and place them in the dishwasher rack",
            lambda env: (
                env.dishwasher.check_rack_contact(env, "dish0")
                and env.dishwasher.check_rack_contact(env, "dish1")
            ),
        ),
        SubtaskDef(
            "Close the dishwasher door",
            lambda env: True,
        ),
    ],
    "PackIdenticalLunches": [
        SubtaskDef(
            "Place one vegetable and one meat into the first tupperware",
            lambda env: (
                (
                    any(
                        OU.check_obj_in_receptacle(env, v, "tupperware0")
                        for v in ["vegetable0", "vegetable1"]
                    )
                    and any(
                        OU.check_obj_in_receptacle(env, m, "tupperware0")
                        for m in ["meat0", "meat1"]
                    )
                )
                or (
                    any(
                        OU.check_obj_in_receptacle(env, v, "tupperware1")
                        for v in ["vegetable0", "vegetable1"]
                    )
                    and any(
                        OU.check_obj_in_receptacle(env, m, "tupperware1")
                        for m in ["meat0", "meat1"]
                    )
                )
            ),
        ),
        SubtaskDef(
            "Place the remaining vegetable and meat into the second tupperware",
            lambda env: True,
        ),
    ],
    "PreSoakPan": [
        SubtaskDef(
            "Pick the pan and sponge and place them both into the sink",
            lambda env: (
                OU.obj_inside_of(env, "obj1", env.sink, partial_check=False)
                and OU.obj_inside_of(env, "obj2", env.sink, partial_check=False)
            ),
        ),
        SubtaskDef(
            "Turn on the water faucet",
            lambda env: True,
        ),
    ],
    "PrepareCoffee": [
        SubtaskDef(
            "Pick the mug from the cabinet and place it under the coffee machine dispenser",
            lambda env: env.coffee_machine.check_receptacle_placement_for_pouring(
                env, "obj"
            ),
        ),
        SubtaskDef(
            "Press the start button on the coffee machine",
            lambda env: True,
        ),
    ],
    "RinseSinkBasin": [
        SubtaskDef(
            "Turn on the sink faucet",
            lambda env: env.sink.get_handle_state(env=env)["water_on"],
        ),
        SubtaskDef(
            "Move the spout to rinse all locations (left, center, right) of the sink basin",
            lambda env: True,
        ),
    ],
    "ScrubCuttingBoard": [
        SubtaskDef(
            "Pick up the sponge from the counter",
            lambda env: OU.check_obj_grasped(env, "sponge"),
        ),
        SubtaskDef(
            "Scrub the cutting board by pressing the sponge across it in multiple locations, then release",
            lambda env: True,
        ),
    ],
    "SearingMeat": [
        SubtaskDef(
            "Grab the pan from the cabinet and place it on the correct burner on the stove",
            lambda env: (
                env.stove.check_obj_location_on_stove(env, "pan", threshold=0.15)
                == env.knob
            ),
        ),
        SubtaskDef(
            "Place the meat on the pan",
            lambda env: OU.check_obj_in_receptacle(env, "meat", "pan", th=0.07),
        ),
        SubtaskDef(
            "Turn the burner on to start searing",
            lambda env: True,
        ),
    ],
    "SetUpCuttingStation": [
        SubtaskDef(
            "Pick up the knife from the drawer and place it on the cutting board",
            lambda env: OU.check_obj_in_receptacle(env, "knife", "receptacle"),
        ),
        SubtaskDef(
            "Move the meat from the plate onto the cutting board",
            lambda env: True,
        ),
    ],
    "StackBowlsCabinet": [
        SubtaskDef(
            "Stack the smaller bowl on top of the larger bowl",
            lambda env: (
                OU.check_obj_in_receptacle(env, "bowl2", "bowl1")
                or OU.check_obj_in_receptacle(env, "bowl1", "bowl2")
            ),
        ),
        SubtaskDef(
            "Move the stacked bowls into the open cabinet",
            lambda env: True,
        ),
    ],
    "SteamInMicrowave": [
        SubtaskDef(
            "Pick the vegetable from the sink and place it in the bowl",
            lambda env: OU.check_obj_in_receptacle(env, "vegetable", "bowl"),
        ),
        SubtaskDef(
            "Pick the bowl and place it inside the microwave",
            lambda env: OU.obj_inside_of(env, "bowl", env.microwave),
        ),
        SubtaskDef(
            "Close the microwave door and press the start button",
            lambda env: True,
        ),
    ],
    "StirVegetables": [
        SubtaskDef(
            "Pick up the vegetables and place them in the pot on the stove",
            lambda env: (
                OU.check_obj_in_receptacle(env, "veg1", "pot")
                and OU.check_obj_in_receptacle(env, "veg2", "pot")
            ),
        ),
        SubtaskDef(
            "Retrieve the spatula and use it to stir the vegetables in the pot",
            lambda env: True,
        ),
    ],
    "StoreLeftoversInBowl": [
        SubtaskDef(
            "Pick the chicken drumstick and vegetable from their plates and place them in the bowl",
            lambda env: (
                OU.check_obj_in_receptacle(env, "chicken_drumstick", "bowl")
                and OU.check_obj_in_receptacle(env, "vegetable", "bowl")
            ),
        ),
        SubtaskDef(
            "Put the bowl with leftovers into the fridge",
            lambda env: True,
        ),
    ],
    "WashLettuce": [
        SubtaskDef(
            "Turn on the sink faucet",
            lambda env: env.sink.get_handle_state(env=env)["water_on"],
        ),
        SubtaskDef(
            "Hold the lettuce under the running water to wash it",
            lambda env: True,
        ),
    ],
}


class SubtaskContextWrapper(gym.Wrapper):
    """Appends ground-truth subtask progress context to the language instruction
    at every step, for studying how much progress context improves GR00T on
    composite long-horizon tasks.

    The injected string has the form:
        "{original instruction} [Progress: subtask {n}/{total} — {description}]"

    Usage (in _create_single_env):
        env = gym.make(...)
        env = SubtaskContextWrapper(env)
        env = VideoRecordingWrapper(env, ...)
        env = MultiStepWrapper(env, ...)
    """

    LANG_KEY = "annotation.human.task_description"

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._subtasks: Optional[List[SubtaskDef]] = None
        self._base_lang: str = ""

    def _get_kitchen_env(self):
        """Fully unwrap to the underlying Kitchen task instance."""
        e = self.env
        while hasattr(e, "env"):
            e = e.env
        return e

    def _resolve_task(self):
        """Identify the current task class and cache its subtask list."""
        kitchen = self._get_kitchen_env()
        name = type(kitchen).__name__
        self._subtasks = SUBTASK_REGISTRY.get(name)  # None if task not in registry

    def _current_subtask_idx(self) -> int:
        """Return the 0-based index of the first incomplete subtask."""
        kitchen = self._get_kitchen_env()
        for i, st in enumerate(self._subtasks[:-1]):  # last entry is always terminal
            try:
                if not st.is_complete(kitchen):
                    return i
            except Exception:
                return i
        return len(self._subtasks) - 1

    def _augment_obs(self, obs: dict) -> dict:
        if self._subtasks is None:
            return obs
        idx = self._current_subtask_idx()
        total = len(self._subtasks)
        desc = self._subtasks[idx].description
        obs = dict(obs)
        obs[self.LANG_KEY] = (
            self._base_lang + f" [Progress: subtask {idx + 1}/{total} — {desc}]"
        )
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._resolve_task()
        self._base_lang = obs.get(self.LANG_KEY, "")
        obs = self._augment_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._augment_obs(obs)
        return obs, reward, terminated, truncated, info
