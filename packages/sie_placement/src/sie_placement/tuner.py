"""One tick of the online lane tuner, kept inside the envelope a plan writes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from sie_placement.schema import Envelope


class Action(StrEnum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD = "hold"
    HOLD_AT_CAP = "hold_at_cap"
    RAISE_TO_FLOOR = "raise_to_floor"


class Attainment(StrEnum):
    MET = "met"
    BREACHED = "breached"
    UNMEASURED = "unmeasured"


def _require_non_negative(name: str, value: int | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be at least 0, got {value!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Observation:
    runners: int
    backlog: int
    running: int
    attainment: Attainment = Attainment.UNMEASURED

    def __post_init__(self) -> None:
        _require_non_negative("runners", self.runners)
        _require_non_negative("backlog", self.backlog)
        _require_non_negative("running", self.running)
        object.__setattr__(self, "attainment", Attainment(self.attainment))

    @property
    def pressured(self) -> bool:
        return self.backlog > 0 or self.attainment is Attainment.BREACHED

    @property
    def idle(self) -> bool:
        return not self.pressured and self.running == 0


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    floor: int
    buffer: int | None = None
    scaledown_window_s: int | None = None

    def __post_init__(self) -> None:
        _require_non_negative("floor", self.floor)
        _require_non_negative("buffer", self.buffer)
        _require_non_negative("scaledown_window_s", self.scaledown_window_s)


@dataclass(frozen=True, slots=True, kw_only=True)
class Step:
    action: Action
    settings: Settings


def _one_toward(value: int, target: int) -> int:
    if value < target:
        return value + 1
    if value > target:
        return value - 1
    return value


def _halfway_toward(value: int, target: int) -> int:
    if value < target:
        return value + (target - value + 1) // 2
    return value - (value - target + 1) // 2


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _optional(move: Callable[[int, int], int], value: int | None, target: int | None) -> int | None:
    if value is None or target is None:
        return value
    return move(value, target)


def _optional_clamp(value: int | None, lower: int | None, upper: int | None) -> int | None:
    if value is None or lower is None or upper is None:
        return value
    return _clamp(value, lower, upper)


def _edge(envelope: Envelope, current: Settings, *, upper: bool) -> Settings:
    buffer_end = envelope.buffer_max if upper else envelope.buffer_min
    window_end = envelope.scaledown_window_max_s if upper else envelope.scaledown_window_min_s
    return Settings(
        floor=envelope.floor_max if upper else envelope.floor_min,
        buffer=current.buffer if current.buffer is None or buffer_end is None else buffer_end,
        scaledown_window_s=(
            current.scaledown_window_s if current.scaledown_window_s is None or window_end is None else window_end
        ),
    )


def _toward(settings: Settings, edge: Settings) -> Settings:
    return Settings(
        floor=_one_toward(settings.floor, edge.floor),
        buffer=_optional(_one_toward, settings.buffer, edge.buffer),
        scaledown_window_s=_optional(_halfway_toward, settings.scaledown_window_s, edge.scaledown_window_s),
    )


def step(envelope: Envelope, observation: Observation, current: Settings) -> Step:
    lowest = _edge(envelope, current, upper=False)
    highest = _edge(envelope, current, upper=True)
    inside = Settings(
        floor=_clamp(current.floor, lowest.floor, highest.floor),
        buffer=_optional_clamp(current.buffer, lowest.buffer, highest.buffer),
        scaledown_window_s=_optional_clamp(
            current.scaledown_window_s, lowest.scaledown_window_s, highest.scaledown_window_s
        ),
    )
    if current.floor < envelope.floor_min:
        return Step(action=Action.RAISE_TO_FLOOR, settings=inside)
    if current.floor > envelope.floor_max:
        return Step(action=Action.HOLD_AT_CAP, settings=inside)
    if observation.pressured:
        raised = _toward(inside, highest)
        if raised == inside:
            return Step(action=Action.HOLD_AT_CAP, settings=inside)
        if observation.runners < inside.floor:
            return Step(action=Action.HOLD, settings=inside)
        return Step(action=Action.SCALE_UP, settings=raised)
    if observation.idle:
        lowered = _toward(inside, lowest)
        if lowered != inside:
            return Step(action=Action.SCALE_DOWN, settings=lowered)
    return Step(action=Action.HOLD, settings=inside)
