from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import pytest
from sie_placement.schema import Envelope
from sie_placement.tuner import Action, Attainment, Observation, Settings, Step, step

FLOOR_RANGES = [(0, 0), (0, 1), (0, 3), (1, 1), (1, 4), (2, 3)]
BUFFER_RANGES: list[tuple[int, int] | None] = [None, (0, 0), (0, 2), (1, 3)]
WINDOW_RANGES: list[tuple[int, int] | None] = [None, (60, 60), (10, 300)]


def make_envelope(
    floor: tuple[int, int],
    buffer: tuple[int, int] | None = None,
    window: tuple[int, int] | None = None,
) -> Envelope:
    return Envelope(
        floor_min=floor[0],
        floor_max=floor[1],
        buffer_min=None if buffer is None else buffer[0],
        buffer_max=None if buffer is None else buffer[1],
        scaledown_window_min_s=None if window is None else window[0],
        scaledown_window_max_s=None if window is None else window[1],
    )


ENVELOPES = [
    make_envelope(floor, buffer, window)
    for floor, buffer, window in itertools.product(FLOOR_RANGES, BUFFER_RANGES, WINDOW_RANGES)
]


def floor_only(envelope: Envelope) -> Envelope:
    return Envelope(floor_min=envelope.floor_min, floor_max=envelope.floor_max)


def around(bounds: tuple[int | None, int | None]) -> list[int | None]:
    lower, upper = bounds
    if lower is None or upper is None:
        return [None, 0, 5]
    values = {lower - 1, lower, (lower + upper) // 2, upper, upper + 1}
    return [None, *sorted(value for value in values if value >= 0)]


def currents(envelope: Envelope) -> Iterator[Settings]:
    for floor, buffer, window in itertools.product(
        range(envelope.floor_max + 2),
        around((envelope.buffer_min, envelope.buffer_max)),
        around((envelope.scaledown_window_min_s, envelope.scaledown_window_max_s)),
    ):
        yield Settings(floor=floor, buffer=buffer, scaledown_window_s=window)


def observations(current: Settings) -> Iterator[Observation]:
    for runners, backlog, running, attainment in itertools.product(
        sorted({max(0, current.floor - 1), current.floor, current.floor + 1}),
        (0, 2),
        (0, 1),
        Attainment,
    ):
        yield Observation(runners=runners, backlog=backlog, running=running, attainment=attainment)


def tuned(lower: int | None, value: int | None) -> bool:
    return lower is not None and value is not None


def bounds_of(envelope: Envelope) -> list[tuple[str, int | None, int | None]]:
    return [
        ("floor", envelope.floor_min, envelope.floor_max),
        ("buffer", envelope.buffer_min, envelope.buffer_max),
        ("scaledown_window_s", envelope.scaledown_window_min_s, envelope.scaledown_window_max_s),
    ]


def clamped(envelope: Envelope, current: Settings) -> Settings:
    values: dict[str, int | None] = {}
    for name, lower, upper in bounds_of(envelope):
        value = getattr(current, name)
        if lower is None or upper is None or value is None:
            values[name] = value
        else:
            values[name] = max(lower, min(value, upper))
    return Settings(**values)


def tuned_pairs(envelope: Envelope, current: Settings, result: Settings) -> list[tuple[int, int, int, int]]:
    pairs = []
    for name, lower, upper in bounds_of(envelope):
        before = getattr(current, name)
        if lower is None or upper is None or before is None:
            continue
        pairs.append((before, getattr(result, name), lower, upper))
    return pairs


def assert_step_invariants(envelope: Envelope, current: Settings, observation: Observation, result: Step) -> None:
    settings = result.settings
    assert envelope.floor_min <= settings.floor <= envelope.floor_max
    for name, lower, upper in bounds_of(envelope)[1:]:
        before, after = getattr(current, name), getattr(settings, name)
        if tuned(lower, before):
            assert lower is not None
            assert upper is not None
            assert after is not None
            assert lower <= after <= upper
        else:
            assert after == before

    idle = observation.backlog == 0 and observation.running == 0 and observation.attainment is not Attainment.BREACHED
    pressured = observation.backlog > 0 or observation.attainment is Attainment.BREACHED
    inside = clamped(envelope, current)
    pairs = tuned_pairs(envelope, current, settings)
    in_envelope = inside == current

    if current.floor < envelope.floor_min:
        assert result.action is Action.RAISE_TO_FLOOR
        assert settings == inside
        assert settings.floor == envelope.floor_min
        return
    if current.floor > envelope.floor_max:
        assert result.action is Action.HOLD_AT_CAP
        assert settings == inside
        assert settings.floor == envelope.floor_max
        return

    assert abs(settings.floor - current.floor) <= 1
    if result.action is Action.SCALE_UP:
        assert pressured
        assert observation.runners >= current.floor
        assert all(after >= min(max(before, lower), upper) for before, after, lower, upper in pairs)
        assert settings != inside
    elif result.action is Action.SCALE_DOWN:
        assert idle
        assert all(after <= min(max(before, lower), upper) for before, after, lower, upper in pairs)
        assert settings != inside
    elif result.action is Action.HOLD_AT_CAP:
        assert pressured
        assert settings == inside
        assert all(after == upper for _, after, _, upper in pairs)
    elif result.action is Action.HOLD:
        assert settings == inside
    else:
        pytest.fail(f"unexpected {result.action} for a floor inside the envelope")

    if in_envelope and result.action in {Action.HOLD, Action.HOLD_AT_CAP}:
        assert settings == current
    if not idle:
        assert all(after >= min(max(before, lower), upper) for before, after, lower, upper in pairs)
    if not pressured:
        assert all(after <= min(max(before, lower), upper) for before, after, lower, upper in pairs)

    room_above = any(min(max(before, lower), upper) < upper for before, _, lower, upper in pairs)
    room_below = any(min(max(before, lower), upper) > lower for before, _, lower, upper in pairs)
    if pressured and not room_above:
        assert result.action is Action.HOLD_AT_CAP
    if pressured and room_above and observation.runners >= current.floor:
        assert result.action is Action.SCALE_UP
    if idle and room_below:
        assert result.action is Action.SCALE_DOWN


@pytest.mark.parametrize("envelope", ENVELOPES, ids=lambda envelope: repr(envelope.to_dict()))
def test_every_step_over_the_grid_stays_inside_the_envelope(envelope):
    for current in currents(envelope):
        for observation in observations(current):
            result = step(envelope, observation, current)
            assert_step_invariants(envelope, current, observation, result)

            floor_result = step(floor_only(envelope), observation, Settings(floor=current.floor))
            assert result.settings.floor == floor_result.settings.floor
            assert floor_result.settings == Settings(floor=floor_result.settings.floor)

            if current.buffer is None and current.scaledown_window_s is None:
                assert result == step(floor_only(envelope), observation, current)


@pytest.mark.parametrize("floor", [0, 1, 3])
def test_raise_to_floor_restores_a_floor_reset_below_the_envelope(floor):
    envelope = make_envelope((4, 6), (1, 2), (60, 600))
    current = Settings(floor=floor, buffer=0, scaledown_window_s=30)
    for observation in observations(current):
        assert step(envelope, observation, current) == Step(
            action=Action.RAISE_TO_FLOOR, settings=Settings(floor=4, buffer=1, scaledown_window_s=60)
        )


def test_a_floor_above_the_envelope_is_held_at_the_cap():
    envelope = make_envelope((1, 3))
    idle = Observation(runners=5, backlog=0, running=0)
    assert step(envelope, idle, Settings(floor=5)) == Step(action=Action.HOLD_AT_CAP, settings=Settings(floor=3))


def test_hold_at_cap_applies_nothing_at_the_edge():
    envelope = make_envelope((1, 3), (0, 2), (60, 600))
    current = Settings(floor=3, buffer=2, scaledown_window_s=600)
    for observation in (
        Observation(runners=3, backlog=4, running=3),
        Observation(runners=3, backlog=0, running=0, attainment=Attainment.BREACHED),
        Observation(runners=1, backlog=4, running=1),
    ):
        assert step(envelope, observation, current) == Step(action=Action.HOLD_AT_CAP, settings=current)


def test_scale_up_moves_every_tuned_setting_one_step():
    envelope = make_envelope((0, 4), (0, 3), (60, 900))
    current = Settings(floor=1, buffer=1, scaledown_window_s=60)
    observation = Observation(runners=1, backlog=3, running=1)
    assert step(envelope, observation, current) == Step(
        action=Action.SCALE_UP, settings=Settings(floor=2, buffer=2, scaledown_window_s=480)
    )


def test_scale_down_happens_only_when_the_lane_is_idle():
    envelope = make_envelope((0, 4), (0, 3), (60, 900))
    current = Settings(floor=3, buffer=2, scaledown_window_s=900)
    busy = Observation(runners=3, backlog=0, running=2)
    queued = Observation(runners=3, backlog=1, running=0)
    breached = Observation(runners=3, backlog=0, running=0, attainment=Attainment.BREACHED)
    idle = Observation(runners=3, backlog=0, running=0)
    assert step(envelope, busy, current) == Step(action=Action.HOLD, settings=current)
    assert step(envelope, queued, current).action is Action.SCALE_UP
    assert step(envelope, breached, current).action is Action.SCALE_UP
    assert step(envelope, idle, current) == Step(
        action=Action.SCALE_DOWN, settings=Settings(floor=2, buffer=1, scaledown_window_s=480)
    )


def test_an_idle_lane_at_the_envelope_minimum_holds():
    envelope = make_envelope((1, 4), (0, 3), (60, 900))
    current = Settings(floor=1, buffer=0, scaledown_window_s=60)
    idle = Observation(runners=1, backlog=0, running=0)
    assert step(envelope, idle, current) == Step(action=Action.HOLD, settings=current)


def test_a_boot_in_progress_is_not_answered_with_another_raise():
    envelope = make_envelope((0, 8))
    current = Settings(floor=2)
    booting = Observation(runners=1, backlog=5, running=1)
    assert step(envelope, booting, current) == Step(action=Action.HOLD, settings=current)


@pytest.mark.parametrize("attainment", [Attainment.MET, Attainment.UNMEASURED])
def test_a_service_level_that_is_not_breached_steers_on_the_queue_alone(attainment):
    envelope = make_envelope((0, 4))
    current = Settings(floor=2)
    for runners, backlog, running in itertools.product(range(4), range(3), range(3)):
        with_attainment = Observation(runners=runners, backlog=backlog, running=running, attainment=attainment)
        queue_only = Observation(runners=runners, backlog=backlog, running=running)
        assert step(envelope, with_attainment, current) == step(envelope, queue_only, current)


def test_a_breached_service_level_raises_an_empty_lane():
    envelope = make_envelope((0, 2))
    breached = Observation(runners=0, backlog=0, running=0, attainment=Attainment.BREACHED)
    assert step(envelope, breached, Settings(floor=0)) == Step(action=Action.SCALE_UP, settings=Settings(floor=1))


def test_the_window_moves_halfway_to_the_edge_of_its_range():
    envelope = make_envelope((0, 0), window=(60, 900))
    pressure = Observation(runners=0, backlog=1, running=0)
    idle = Observation(runners=0, backlog=0, running=0)
    current = Settings(floor=0, scaledown_window_s=60)
    rising = []
    while (result := step(envelope, pressure, current)).action is Action.SCALE_UP:
        current = result.settings
        rising.append(current.scaledown_window_s)
    falling = []
    while (result := step(envelope, idle, current)).action is Action.SCALE_DOWN:
        current = result.settings
        falling.append(current.scaledown_window_s)
    assert rising == [480, 690, 795, 848, 874, 887, 894, 897, 899, 900]
    assert falling == [480, 270, 165, 112, 86, 73, 66, 63, 61, 60]


def test_untuned_settings_pass_through_unchanged():
    envelope = make_envelope((0, 2))
    current = Settings(floor=1, buffer=7, scaledown_window_s=1200)
    pressure = Observation(runners=1, backlog=2, running=1)
    assert step(envelope, pressure, current) == Step(
        action=Action.SCALE_UP, settings=Settings(floor=2, buffer=7, scaledown_window_s=1200)
    )


def observe(settings: Settings, demand: int, runners: int | None = None) -> Observation:
    active = demand > 0
    capacity = settings.floor + ((settings.buffer or 0) if active else 0)
    started = capacity if runners is None else runners
    running = min(demand, started)
    return Observation(runners=started, backlog=demand - running, running=running)


def run(envelope: Envelope, start: Settings, demand: int, ticks: int) -> list[Step]:
    steps = []
    current = start
    for _ in range(ticks):
        result = step(envelope, observe(current, demand), current)
        steps.append(result)
        current = result.settings
    return steps


def ticks_to_settle(steps: list[Step]) -> int:
    for index, result in enumerate(steps):
        if result.action in {Action.HOLD, Action.HOLD_AT_CAP}:
            return index
    pytest.fail("the tuner did not settle")


def minimum(envelope: Envelope) -> Settings:
    return Settings(
        floor=envelope.floor_min,
        buffer=envelope.buffer_min,
        scaledown_window_s=envelope.scaledown_window_min_s,
    )


@pytest.mark.parametrize("demand", [0, 1, 3, 4, 6, 9])
@pytest.mark.parametrize("start", [1, 2, 5, 6])
def test_a_reset_to_the_envelope_minimum_reconverges_to_the_floor_demand_needs(demand, start):
    envelope = make_envelope((1, 6))
    before = run(envelope, Settings(floor=start), demand, 12)
    settled = before[-1].settings
    assert before[-1].action in {Action.HOLD, Action.HOLD_AT_CAP}

    after = run(envelope, minimum(envelope), demand, 12)
    needed = max(envelope.floor_min, min(demand, envelope.floor_max))
    assert ticks_to_settle(after) <= envelope.floor_max - envelope.floor_min
    assert after[-1].settings == Settings(floor=needed)
    if demand > envelope.floor_max:
        assert after[-1].action is Action.HOLD_AT_CAP
    if start <= needed:
        assert settled == after[-1].settings
    else:
        assert settled.floor >= needed


@pytest.mark.parametrize("envelope", ENVELOPES, ids=lambda envelope: repr(envelope.to_dict()))
def test_sustained_pressure_from_the_minimum_reaches_the_cap_within_the_bound(envelope):
    start = minimum(envelope)
    pressure = Observation(runners=envelope.floor_max, backlog=1, running=envelope.floor_max)
    bound = max(
        envelope.floor_max - envelope.floor_min,
        0 if envelope.buffer_max is None or envelope.buffer_min is None else envelope.buffer_max - envelope.buffer_min,
        0
        if envelope.scaledown_window_max_s is None or envelope.scaledown_window_min_s is None
        else (envelope.scaledown_window_max_s - envelope.scaledown_window_min_s).bit_length(),
    )
    current = start
    for _ in range(bound):
        result = step(envelope, pressure, current)
        assert result.action is Action.SCALE_UP
        current = result.settings
    assert step(envelope, pressure, current) == Step(action=Action.HOLD_AT_CAP, settings=current)

    idle = Observation(runners=current.floor, backlog=0, running=0)
    for _ in range(bound):
        result = step(envelope, idle, current)
        assert result.action is Action.SCALE_DOWN
        current = result.settings
    assert step(envelope, idle, current) == Step(action=Action.HOLD, settings=start)


@pytest.mark.parametrize("demand", [0, 1, 4, 7, 30])
def test_a_reset_of_every_tuned_setting_reconverges_to_a_fixed_point_that_serves_demand(demand):
    envelope = make_envelope((1, 4), (0, 3), (60, 900))
    bound = max(3, 3, (900 - 60).bit_length())
    after = run(envelope, minimum(envelope), demand, bound + 2)
    assert ticks_to_settle(after) <= bound
    settled = after[-1].settings
    observation = observe(settled, demand)
    result = step(envelope, observation, settled)
    assert result.settings == settled
    assert observation.backlog == 0 or result.action is Action.HOLD_AT_CAP


def test_a_slow_boot_raises_the_floor_once_per_realized_step():
    envelope = make_envelope((0, 8))
    current = Settings(floor=0)
    boot_ticks = 3
    history = [0]
    for _ in range(30):
        runners = history[max(0, len(history) - 1 - boot_ticks)]
        result = step(envelope, observe(current, 5, runners), current)
        assert result.settings.floor <= runners + 1
        current = result.settings
        history.append(current.floor)
    assert current.floor == 5


@pytest.mark.parametrize(
    "make",
    [
        lambda: Observation(runners=-1, backlog=0, running=0),
        lambda: Observation(runners=0, backlog=-1, running=0),
        lambda: Observation(runners=0, backlog=0, running=-1),
        lambda: Settings(floor=-1),
        lambda: Settings(floor=0, buffer=-1),
        lambda: Settings(floor=0, scaledown_window_s=-1),
    ],
)
def test_negative_counts_are_rejected(make):
    with pytest.raises(ValueError, match="must be at least 0"):
        make()


@pytest.mark.parametrize("attainment", list(Attainment))
def test_an_attainment_given_as_its_wire_string_reads_as_its_value(attainment):
    wire: Any = attainment.value
    envelope = make_envelope((0, 4))
    for runners, backlog, running in itertools.product(range(2), range(2), range(2)):
        from_wire = Observation(runners=runners, backlog=backlog, running=running, attainment=wire)
        typed = Observation(runners=runners, backlog=backlog, running=running, attainment=attainment)
        assert from_wire.attainment is attainment
        assert from_wire == typed
        assert step(envelope, from_wire, Settings(floor=1)) == step(envelope, typed, Settings(floor=1))


def test_an_unknown_attainment_is_rejected():
    wire: Any = "late"
    with pytest.raises(ValueError, match="not a valid Attainment"):
        Observation(runners=0, backlog=0, running=0, attainment=wire)


def test_actions_carry_their_wire_names():
    assert [action.value for action in Action] == ["scale_up", "scale_down", "hold", "hold_at_cap", "raise_to_floor"]
    assert [attainment.value for attainment in Attainment] == ["met", "breached", "unmeasured"]
