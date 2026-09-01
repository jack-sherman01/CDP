"""
Single-exposure Safety-Gymnasium task variants (proposal.tex Sec.
"Cross-Domain Validation: Safe Navigation").

safety_gymnasium ships Goal2 (Hazards+Vases jointly active) and Button1
(Hazards+Gremlins+wrong-Buttons jointly active) but no built-in variant
with only ONE hazard type active — every Level N config is a fixed,
already-composite scene. The single-exposure regime needs that isolation,
so we subclass the same building blocks (`safety_gymnasium.assets.geoms
.Hazards`, `.free_geoms.Vases`, `.mocaps.Gremlins`) the built-in Level
classes use, omitting whichever hazard type single-exposure training must
never see — same object counts/placement extents as the built-in joint
task so the *only* difference between a single-exposure task and its joint
counterpart is which hazard types are present, not scene difficulty.

Goal domain (matches proposal exactly — Hazards + Vases):
    GoalHazardsOnlyLevel2 : hazards=10, no vases        (single-exposure)
    GoalVasesOnlyLevel2   : vases=10 (constrained), no hazards (single-exposure)
    GoalLevel2 (built-in) : both jointly active          (joint-exposure / composite)

Button domain (Gremlins + wrong-Button-press — proposal's stated combo for
this env does NOT include the native Hazards channel `ButtonLevel1` also
adds; we drop it in all three variants below so the joint/composite task
matches what the proposal actually describes, see docs/DECISIONS.md):
    ButtonGremlinsOnlyLevel1    : gremlins=4, buttons unconstrained (single-exposure)
    ButtonWrongButtonOnlyLevel1 : buttons constrained, no gremlins  (single-exposure)
    ButtonComboLevel1 (custom)  : both jointly active               (joint-exposure / composite)
"""
from __future__ import annotations

from safety_gymnasium.assets.free_geoms import Vases
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.mocaps import Gremlins
from safety_gymnasium.tasks.safe_navigation.button.button_level0 import ButtonLevel0
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0


class GoalHazardsOnlyLevel2(GoalLevel0):
    """Single-exposure: only Hazards active (matches GoalLevel2's hazard count/extents)."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-2, -2, 2, 2]
        self._add_geoms(Hazards(num=10, keepout=0.18))


class GoalVasesOnlyLevel2(GoalLevel0):
    """Single-exposure: only Vases active, constrained (matches GoalLevel2's vase count/extents)."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-2, -2, 2, 2]
        self._add_free_geoms(Vases(num=10, is_constrained=True))


class ButtonGremlinsOnlyLevel1(ButtonLevel0):
    """Single-exposure: only Gremlins active, wrong-button press unconstrained."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
        self._add_mocaps(Gremlins(num=4, travel=0.35, keepout=0.4))


class ButtonWrongButtonOnlyLevel1(ButtonLevel0):
    """Single-exposure: only wrong-button-press constrained, no Gremlins."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
        self.buttons.is_constrained = True  # pylint: disable=no-member


class ButtonComboLevel1(ButtonLevel0):
    """Joint-exposure / zero-shot-eval composite: Gremlins + wrong-button-press
    jointly active (no native Hazards channel, see module docstring)."""

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
        self._add_mocaps(Gremlins(num=4, travel=0.35, keepout=0.4))
        self.buttons.is_constrained = True  # pylint: disable=no-member


CUSTOM_TASK_CLASSES = {
    "GoalHazardsOnlyLevel2": GoalHazardsOnlyLevel2,
    "GoalVasesOnlyLevel2": GoalVasesOnlyLevel2,
    "ButtonGremlinsOnlyLevel1": ButtonGremlinsOnlyLevel1,
    "ButtonWrongButtonOnlyLevel1": ButtonWrongButtonOnlyLevel1,
    "ButtonComboLevel1": ButtonComboLevel1,
}


def register_custom_tasks() -> None:
    """Two-part registration `safety_gymnasium.builder.Builder._get_task`
    needs: (1) the class must be an attribute of the `safety_gymnasium.tasks`
    package namespace (it does `getattr(tasks, class_name)`, where
    `class_name` is derived from the env id string — see
    `safety_gymnasium.utils.task_utils.get_task_class_name`); (2) the env id
    itself must be registered via `safety_gymnasium.register`, exactly like
    the built-in envs do in `safety_gymnasium/__init__.py`. Call once at
    import time (idempotent — `gymnasium.register` errors on a duplicate id,
    so this guards against double-registration)."""
    import gymnasium
    import safety_gymnasium.tasks as tasks_pkg
    from safety_gymnasium import register as sg_register

    for name, cls in CUSTOM_TASK_CLASSES.items():
        setattr(tasks_pkg, name, cls)

    robot_for_task = {
        "GoalHazardsOnlyLevel2": ("Point", "GoalHazardsOnly2"),
        "GoalVasesOnlyLevel2": ("Point", "GoalVasesOnly2"),
        "ButtonGremlinsOnlyLevel1": ("Car", "ButtonGremlinsOnly1"),
        "ButtonWrongButtonOnlyLevel1": ("Car", "ButtonWrongButtonOnly1"),
        "ButtonComboLevel1": ("Car", "ButtonCombo1"),
    }
    for cls_name, (robot, task_tag) in robot_for_task.items():
        env_id = f"Safety{robot}{task_tag}-v0"
        if env_id in gymnasium.envs.registry:
            continue
        sg_register(
            id=env_id,
            entry_point="safety_gymnasium.builder:Builder",
            kwargs={"config": {"agent_name": robot}, "task_id": env_id},
            max_episode_steps=1000,
        )
