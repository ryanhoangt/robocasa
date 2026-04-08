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
    # paper: 4 | tracked: 3 (close-drawer after picking undetectable)
    "DeliverStraw": [
        SubtaskDef(
            "Open the drawer",
            lambda env: env.drawer.is_open(env),
        ),
        SubtaskDef(
            "Pick up the straw from the drawer",
            lambda env: OU.check_obj_grasped(env, "straw")
            or OU.check_obj_in_receptacle(env, "straw", "glass_cup", th=0.5),
        ),
        SubtaskDef(
            "Place the straw inside the glass cup on the dining counter",
            lambda env: OU.check_obj_in_receptacle(env, "straw", "glass_cup", th=0.5),
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 4 | tracked: 3 (passive wait for lever-pop is undetectable as a separate step)
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
            lambda env: OU.check_obj_in_receptacle(env, "obj", "plate"),
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 2 | tracked: 2 (exact match)
    "KettleBoiling": [
        SubtaskDef(
            "Pick up the kettle from the counter and place it on a stove burner",
            lambda env: OU.check_obj_fixture_contact(env, "obj", env.stove),
        ),
        SubtaskDef(
            "Turn the burner on to start boiling",
            lambda env: env.stove.check_obj_location_on_stove(
                env=env, obj_name="obj", threshold=0.15
            )
            is not None,
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 3 | tracked: 3 (exact match)
    "LoadDishwasher": [
        SubtaskDef(
            "Place the first dish into the dishwasher rack",
            lambda env: env.dishwasher.check_rack_contact(env, "dish0"),
        ),
        SubtaskDef(
            "Place the second dish into the dishwasher rack",
            lambda env: (
                env.dishwasher.check_rack_contact(env, "dish0")
                and env.dishwasher.check_rack_contact(env, "dish1")
            ),
        ),
        SubtaskDef(
            "Close the dishwasher door",
            lambda env: env.dishwasher.is_closed(env, th=0.05),
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 15 | tracked: 3 (paper counts individual pick+place ops; we track object placements)
    "PackIdenticalLunches": [
        SubtaskDef(
            "Place at least one vegetable into a tupperware",
            lambda env: any(
                OU.check_obj_in_receptacle(env, v, t)
                for v in ["vegetable0", "vegetable1"]
                for t in ["tupperware0", "tupperware1"]
            ),
        ),
        SubtaskDef(
            "Place both vegetables into the tupperwares",
            lambda env: (
                any(
                    OU.check_obj_in_receptacle(env, "vegetable0", t)
                    for t in ["tupperware0", "tupperware1"]
                )
                and any(
                    OU.check_obj_in_receptacle(env, "vegetable1", t)
                    for t in ["tupperware0", "tupperware1"]
                )
            ),
        ),
        SubtaskDef(
            "Place at least one meat into a tupperware",
            lambda env: any(
                OU.check_obj_in_receptacle(env, m, t)
                for m in ["meat0", "meat1"]
                for t in ["tupperware0", "tupperware1"]
            ),
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 3 | tracked: 3 (exact match)
    "PreSoakPan": [
        SubtaskDef(
            "Place the pan into the sink",
            lambda env: OU.obj_inside_of(env, "obj1", env.sink, partial_check=False),
        ),
        SubtaskDef(
            "Place the sponge into the sink",
            lambda env: (
                OU.obj_inside_of(env, "obj1", env.sink, partial_check=False)
                and OU.obj_inside_of(env, "obj2", env.sink, partial_check=False)
            ),
        ),
        SubtaskDef(
            "Turn on the water faucet",
            lambda env: env.sink.get_handle_state(env=env)["water_on"],
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 2 | tracked: 2 (exact match)
    "PrepareCoffee": [
        SubtaskDef(
            "Pick the mug from the cabinet and place it under the coffee machine dispenser",
            lambda env: env.coffee_machine.check_receptacle_placement_for_pouring(
                env, "obj"
            ),
        ),
        SubtaskDef(
            "Press the start button on the coffee machine",
            lambda env: env.coffee_machine._turned_on,
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 2 | tracked: 2 (exact match)
    "RinseSinkBasin": [
        SubtaskDef(
            "Turn on the sink faucet",
            lambda env: env.sink.get_handle_state(env=env)["water_on"],
        ),
        SubtaskDef(
            "Move the spout to rinse all locations (left, center, right) of the sink basin",
            lambda env: all(env.washed_loc),
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 2 | tracked: 2 (exact match)
    "ScrubCuttingBoard": [
        SubtaskDef(
            "Pick up the sponge from the counter",
            lambda env: OU.check_obj_grasped(env, "sponge"),
        ),
        SubtaskDef(
            "Scrub the cutting board across multiple locations, then release",
            lambda env: env.board_contact_timer >= 5,
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 3 | tracked: 3 (exact match; split pan-on-stove from pan-on-correct-lit-burner)
    "SearingMeat": [
        SubtaskDef(
            "Grab the pan from the cabinet and place it on the stove",
            lambda env: OU.check_obj_fixture_contact(env, "pan", env.stove),
        ),
        SubtaskDef(
            "Position the pan on the correct burner and turn it on",
            lambda env: (
                env.stove.check_obj_location_on_stove(
                    env=env, obj_name="pan", threshold=0.15
                )
                == env.knob
            ),
        ),
        SubtaskDef(
            "Place the meat on the pan to start searing",
            lambda env: OU.check_obj_in_receptacle(env, "meat", "pan", th=0.07),
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 2 | tracked: 2 (exact match)
    "SetUpCuttingStation": [
        SubtaskDef(
            "Pick up the knife from the drawer and place it on the cutting board",
            lambda env: OU.check_obj_in_receptacle(env, "knife", "receptacle"),
        ),
        SubtaskDef(
            "Move the meat from the plate onto the cutting board",
            lambda env: OU.check_obj_in_receptacle(env, "meat", "receptacle"),
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 2 | tracked: 2 (exact match)
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
            lambda env: (
                OU.obj_inside_of(env, "bowl1", env.cabinet)
                and OU.obj_inside_of(env, "bowl2", env.cabinet)
            ),
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 6 | tracked: 4 (paper counts pick+place separately; pick ops are non-persistent)
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
            "Close the microwave door",
            lambda env: env.microwave.is_closed(env),
        ),
        SubtaskDef(
            "Press the start button on the microwave",
            lambda env: env.microwave.get_state()["turned_on"],
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 4 | tracked: 3 (stirring motion requires time-series detection, not a static state check)
    "StirVegetables": [
        SubtaskDef(
            "Pick up the first vegetable and place it in the pot",
            lambda env: OU.check_obj_in_receptacle(env, "veg1", "pot"),
        ),
        SubtaskDef(
            "Pick up the second vegetable and place it in the pot",
            lambda env: (
                OU.check_obj_in_receptacle(env, "veg1", "pot")
                and OU.check_obj_in_receptacle(env, "veg2", "pot")
            ),
        ),
        SubtaskDef(
            "Retrieve the spatula",
            lambda env: OU.check_obj_grasped(env, "spatula"),
        ),
        SubtaskDef("Stir the vegetables in the pot", lambda env: True),
    ],
    # paper: 5 | tracked: 3 (paper counts individual pick+place ops)
    "StoreLeftoversInBowl": [
        SubtaskDef(
            "Place the chicken drumstick into the bowl",
            lambda env: OU.check_obj_in_receptacle(env, "chicken_drumstick", "bowl"),
        ),
        SubtaskDef(
            "Place the vegetable into the bowl",
            lambda env: (
                OU.check_obj_in_receptacle(env, "chicken_drumstick", "bowl")
                and OU.check_obj_in_receptacle(env, "vegetable", "bowl")
            ),
        ),
        SubtaskDef(
            "Put the bowl with leftovers into the fridge",
            lambda env: OU.check_obj_fixture_contact(env, "bowl", env.fridge),
        ),
        SubtaskDef("Done", lambda env: True),
    ],
    # paper: 2 | tracked: 2 (exact match)
    "WashLettuce": [
        SubtaskDef(
            "Turn on the sink faucet",
            lambda env: env.sink.get_handle_state(env=env)["water_on"],
        ),
        SubtaskDef(
            "Hold the lettuce under the running water to wash it",
            lambda env: env.washed_time >= 25,
        ),
        SubtaskDef("Done", lambda env: True),
    ],
}


class SubtaskContextWrapper(gym.Wrapper):
    """Tracks ground-truth subtask progress and optionally augments the language
    instruction at every step.

    When augment_language=True (default), the language observation key is
    overwritten with:
        "{original instruction} [Progress: subtask {n}/{total} — {description}]"

    Regardless of augment_language, the info dict returned by step() always
    contains:
        info["subtask_idx"]  — 0-based index of the current subtask
        info["n_subtasks"]   — total subtask count (excluding terminal sentinel)

    Usage (in _create_single_env):
        env = gym.make(...)
        env = SubtaskContextWrapper(env, augment_language=use_subtask_context)
        env = VideoRecordingWrapper(env, ...)
        env = MultiStepWrapper(env, ...)
    """

    LANG_KEY = "annotation.human.task_description"

    def __init__(self, env: gym.Env, augment_language: bool = True):
        super().__init__(env)
        self._subtasks: Optional[List[SubtaskDef]] = None
        self._base_lang: str = ""
        self._augment_language = augment_language

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
        if self._subtasks is None or not self._augment_language:
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
        if self._subtasks is not None:
            info["subtask_idx"] = self._current_subtask_idx()
            info["n_subtasks"] = len(self._subtasks) - 1  # exclude terminal sentinel
        return obs, reward, terminated, truncated, info
