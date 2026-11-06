"""Versioned input and plan documents of the measured-only placement planner."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

SCHEMA_VERSION = 1

SERVICE_LEVELS_KIND = "placement-service-levels"
RECEIPT_KIND = "placement-receipt"
BOOT_EVIDENCE_KIND = "placement-boot-evidence"
PRICE_LIST_KIND = "placement-price-list"
DEMAND_KIND = "placement-demand"
PLAN_KIND = "placement-plan"

ROOT_PATH = "$"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CURRENCY = re.compile(r"[A-Z]{3}")
_HOUR = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):00:00Z", re.ASCII)


class SchemaError(ValueError):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path
        self.message = message


class Operation(StrEnum):
    ENCODE = "encode"
    SCORE = "score"
    EXTRACT = "extract"
    GENERATE = "generate"


class Warmth(StrEnum):
    PINNED = "pinned"
    COLD_WITH_BOUND = "cold-with-bound"
    BRIDGED = "bridged"
    SNAPSHOT_COLD = "snapshot-cold"
    SCHEDULED_WARM = "scheduled-warm"
    LAZY_RESIDENT = "lazy-resident"


BOOT_WARMTHS = frozenset({Warmth.COLD_WITH_BOUND, Warmth.SNAPSHOT_COLD, Warmth.LAZY_RESIDENT})
WARM_FLOOR_WARMTHS = frozenset({Warmth.PINNED, Warmth.LAZY_RESIDENT})


class BootOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed-out"


def _join(path: str, key: str) -> str:
    if not key:
        return path
    if not path:
        return key
    if key.startswith("["):
        return f"{path}{key}"
    return f"{path}.{key}"


def _within(path: str, error: SchemaError) -> SchemaError:
    return SchemaError(_join(path, error.path), error.message)


def _describe(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"boolean {str(value).lower()}"
    if isinstance(value, int | float):
        return f"number {value!r}"
    if isinstance(value, str):
        return f"string {value!r}"
    if isinstance(value, Mapping):
        return "an object"
    if isinstance(value, list | tuple):
        return "an array"
    return type(value).__name__


def _format_hour(value: datetime) -> str:
    return f"{value.year:04d}-{value.month:02d}-{value.day:02d}T{value.hour:02d}:00:00Z"


def _compact(fields: dict[str, object | None]) -> dict[str, object]:
    return {key: value for key, value in fields.items() if value is not None}


class _Reader:
    def __init__(self, data: object, path: str) -> None:
        if not isinstance(data, Mapping):
            raise SchemaError(path, f"expected an object, got {_describe(data)}")
        self._data: dict[str, object] = {}
        for key, value in data.items():
            if not isinstance(key, str):
                raise SchemaError(path, f"expected string keys, got {_describe(key)}")
            self._data[key] = value
        self._path = path
        self._read: set[str] = set()

    def _at(self, key: str) -> str:
        return _join(self._path, key)

    def _required(self, key: str) -> object:
        self._read.add(key)
        if key not in self._data:
            raise SchemaError(self._at(key), "is required")
        value = self._data[key]
        if value is None:
            raise SchemaError(self._at(key), "must not be null")
        return value

    def _optional(self, key: str) -> object | None:
        self._read.add(key)
        if key not in self._data:
            return None
        value = self._data[key]
        if value is None:
            raise SchemaError(self._at(key), "must be omitted rather than null")
        return value

    def _string(self, key: str, value: object) -> str:
        if not isinstance(value, str):
            raise SchemaError(self._at(key), f"expected a string, got {_describe(value)}")
        return value

    def _integer(self, key: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaError(self._at(key), f"expected an integer, got {_describe(value)}")
        return value

    def _number(self, key: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SchemaError(self._at(key), f"expected a number, got {_describe(value)}")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            raise SchemaError(
                self._at(key), "expected a finite number, got an integer beyond the float range"
            ) from None
        if not finite:
            raise SchemaError(self._at(key), f"expected a finite number, got {value!r}")
        return value

    def header(self, kind: str) -> None:
        found = self._string("kind", self._required("kind"))
        if found != kind:
            raise SchemaError(self._at("kind"), f"expected {kind!r}, got {found!r}")
        version = self._integer("schemaVersion", self._required("schemaVersion"))
        if version != SCHEMA_VERSION:
            raise SchemaError(
                self._at("schemaVersion"),
                f"unsupported version {version}; this reader supports version {SCHEMA_VERSION}",
            )

    def string(self, key: str) -> str:
        return self._string(key, self._required(key))

    def optional_string(self, key: str) -> str | None:
        value = self._optional(key)
        return None if value is None else self._string(key, value)

    def integer(self, key: str) -> int:
        return self._integer(key, self._required(key))

    def optional_integer(self, key: str) -> int | None:
        value = self._optional(key)
        return None if value is None else self._integer(key, value)

    def number(self, key: str) -> float:
        return self._number(key, self._required(key))

    def optional_number(self, key: str) -> float | None:
        value = self._optional(key)
        return None if value is None else self._number(key, value)

    def enum[E: StrEnum](self, key: str, kind: type[E]) -> E:
        value = self.string(key)
        try:
            return kind(value)
        except ValueError:
            choices = ", ".join(repr(member.value) for member in kind)
            raise SchemaError(self._at(key), f"expected one of {choices}, got {value!r}") from None

    def hour(self, key: str) -> datetime:
        value = self.string(key)
        match = _HOUR.fullmatch(value)
        if match is None:
            raise SchemaError(self._at(key), f"expected a UTC hour such as '2026-01-01T00:00:00Z', got {value!r}")
        year, month, day, hour = (int(part) for part in match.groups())
        try:
            return datetime(year, month, day, hour, tzinfo=UTC)
        except ValueError as error:
            raise SchemaError(self._at(key), f"is not a valid time: {error}") from None

    def child[T](self, key: str, parse: Callable[[object, str], T]) -> T:
        return parse(self._required(key), self._at(key))

    def optional_child[T](self, key: str, parse: Callable[[object, str], T]) -> T | None:
        value = self._optional(key)
        return None if value is None else parse(value, self._at(key))

    def children[T](self, key: str, parse: Callable[[object, str], T], *, required: bool = True) -> tuple[T, ...]:
        value = self._required(key) if required else self._optional(key)
        if value is None:
            return ()
        if not isinstance(value, list | tuple):
            raise SchemaError(self._at(key), f"expected an array, got {_describe(value)}")
        return tuple(parse(item, f"{self._at(key)}[{index}]") for index, item in enumerate(value))

    def service_level(self) -> ServiceLevel:
        return self.construct(
            ServiceLevel,
            come_alive_p95_s=self.optional_integer("comeAliveP95S"),
            warm_p99_ms=self.optional_integer("warmP99Ms"),
            ttft_p99_ms=self.optional_integer("ttftP99Ms"),
            itl_p99_ms=self.optional_integer("itlP99Ms"),
            queue_wait_p99_ms=self.optional_integer("queueWaitP99Ms"),
        )

    def construct[**P, T](self, factory: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return factory(*args, **kwargs)
        except SchemaError as error:
            raise _within(self._path, error) from None

    def build[**P, T](self, factory: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        unknown = sorted(set(self._data) - self._read)
        if unknown:
            raise SchemaError(self._at(unknown[0]), "is not a field of this object")
        return self.construct(factory, *args, **kwargs)


def _require_identifier(name: str, value: str) -> None:
    if not value or any(character.isspace() for character in value):
        raise SchemaError(name, f"must be a non-empty string without whitespace, got {value!r}")


def _require_finite(name: str, value: float) -> None:
    try:
        finite = math.isfinite(value)
    except OverflowError:
        raise SchemaError(name, "must be a finite number, got an integer beyond the float range") from None
    if not finite:
        raise SchemaError(name, f"must be a finite number, got {value!r}")


def _require_at_least(name: str, value: float | None, minimum: int) -> None:
    if value is None:
        return
    _require_finite(name, value)
    if value < minimum:
        raise SchemaError(name, f"must be at least {minimum}, got {value!r}")


def _require_positive(name: str, value: float | None) -> None:
    if value is None:
        return
    _require_finite(name, value)
    if value <= 0:
        raise SchemaError(name, f"must be greater than 0, got {value!r}")


def _require_ordered(lower_name: str, lower: float | None, upper_name: str, upper: float | None) -> None:
    if lower is not None and upper is not None and lower > upper:
        raise SchemaError(upper_name, f"must be at least {lower_name} ({lower!r}), got {upper!r}")


def _require_pair(first_name: str, first: object | None, second_name: str, second: object | None) -> None:
    if (first is None) != (second is None):
        missing = first_name if first is None else second_name
        raise SchemaError(missing, f"{first_name} and {second_name} are declared together")


def _require_absent(name: str, value: object | None, reason: str) -> None:
    if value is not None:
        raise SchemaError(name, f"must be omitted {reason}")


def _require_unique(collection: str, field: str, keys: Iterable[Hashable]) -> None:
    seen: set[Hashable] = set()
    for index, key in enumerate(keys):
        if key in seen:
            raise SchemaError(f"{collection}[{index}].{field}", f"duplicates {key!r}")
        seen.add(key)


def _require_hour(name: str, value: datetime) -> None:
    offset = value.utcoffset()
    if offset is None or offset.total_seconds() != 0 or value.minute or value.second or value.microsecond:
        raise SchemaError(name, f"must be a UTC hour boundary, got {value.isoformat()}")


def _prefer(override: int | None, inherited: int | None) -> int | None:
    return inherited if override is None else override


@dataclass(frozen=True, slots=True, kw_only=True)
class ServiceLevel:
    come_alive_p95_s: int | None = None
    warm_p99_ms: int | None = None
    ttft_p99_ms: int | None = None
    itl_p99_ms: int | None = None
    queue_wait_p99_ms: int | None = None

    def __post_init__(self) -> None:
        for name, value in self._fields():
            _require_at_least(name, value, 0)
        if self.warm_p99_ms is not None and (self.ttft_p99_ms is not None or self.itl_p99_ms is not None):
            raise SchemaError("warmP99Ms", "must be omitted when ttftP99Ms and itlP99Ms declare a generation identity")

    def _fields(self) -> tuple[tuple[str, int | None], ...]:
        return (
            ("comeAliveP95S", self.come_alive_p95_s),
            ("warmP99Ms", self.warm_p99_ms),
            ("ttftP99Ms", self.ttft_p99_ms),
            ("itlP99Ms", self.itl_p99_ms),
            ("queueWaitP99Ms", self.queue_wait_p99_ms),
        )

    @property
    def is_generation(self) -> bool:
        return self.ttft_p99_ms is not None or self.itl_p99_ms is not None

    def require_complete(self) -> None:
        _require_pair("ttftP99Ms", self.ttft_p99_ms, "itlP99Ms", self.itl_p99_ms)

    def overlay(self, override: ServiceLevel) -> ServiceLevel:
        return ServiceLevel(
            come_alive_p95_s=_prefer(override.come_alive_p95_s, self.come_alive_p95_s),
            warm_p99_ms=_prefer(override.warm_p99_ms, self.warm_p99_ms),
            ttft_p99_ms=_prefer(override.ttft_p99_ms, self.ttft_p99_ms),
            itl_p99_ms=_prefer(override.itl_p99_ms, self.itl_p99_ms),
            queue_wait_p99_ms=_prefer(override.queue_wait_p99_ms, self.queue_wait_p99_ms),
        )

    def to_dict(self) -> dict[str, object]:
        return _compact(dict(self._fields()))


@dataclass(frozen=True, slots=True, kw_only=True)
class ServiceLevelTier:
    name: str
    level: ServiceLevel

    def __post_init__(self) -> None:
        _require_identifier("name", self.name)
        self.level.require_complete()

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(cls, name=reader.string("name"), level=reader.service_level())

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, **self.level.to_dict()}


@dataclass(frozen=True, slots=True, kw_only=True)
class IdentityServiceLevel:
    identity: str
    level: ServiceLevel
    tier: str | None = None

    def __post_init__(self) -> None:
        _require_identifier("identity", self.identity)
        if self.tier is not None:
            _require_identifier("tier", self.tier)

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(
            cls,
            identity=reader.string("identity"),
            tier=reader.optional_string("tier"),
            level=reader.service_level(),
        )

    def to_dict(self) -> dict[str, object]:
        return _compact({"identity": self.identity, "tier": self.tier}) | self.level.to_dict()


@dataclass(frozen=True, slots=True, kw_only=True)
class ServiceLevels:
    identities: tuple[IdentityServiceLevel, ...]
    tiers: tuple[ServiceLevelTier, ...] = ()

    def __post_init__(self) -> None:
        _require_unique("tiers", "name", (tier.name for tier in self.tiers))
        _require_unique("identities", "identity", (entry.identity for entry in self.identities))
        tiers = {tier.name: tier.level for tier in self.tiers}
        for index, entry in enumerate(self.identities):
            if entry.tier is not None and entry.tier not in tiers:
                raise SchemaError(f"identities[{index}].tier", f"names no declared tier: {entry.tier!r}")
            try:
                self._resolve(entry, tiers).require_complete()
            except SchemaError as error:
                raise _within(f"identities[{index}]", error) from None

    @staticmethod
    def _resolve(entry: IdentityServiceLevel, tiers: Mapping[str, ServiceLevel]) -> ServiceLevel:
        if entry.tier is None:
            return entry.level
        return tiers[entry.tier].overlay(entry.level)

    def effective(self, identity: str) -> ServiceLevel:
        tiers = {tier.name: tier.level for tier in self.tiers}
        for entry in self.identities:
            if entry.identity == identity:
                return self._resolve(entry, tiers)
        raise KeyError(identity)

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        reader.header(SERVICE_LEVELS_KIND)
        return reader.build(
            cls,
            tiers=reader.children("tiers", ServiceLevelTier.from_dict, required=False),
            identities=reader.children("identities", IdentityServiceLevel.from_dict),
        )

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {"kind": SERVICE_LEVELS_KIND, "schemaVersion": SCHEMA_VERSION}
        if self.tiers:
            document["tiers"] = [tier.to_dict() for tier in self.tiers]
        document["identities"] = [entry.to_dict() for entry in self.identities]
        return document


@dataclass(frozen=True, slots=True, kw_only=True)
class Cell:
    machine_profile: str
    gpu_product: str

    def __post_init__(self) -> None:
        _require_identifier("machineProfile", self.machine_profile)
        _require_identifier("gpuProduct", self.gpu_product)

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(
            cls,
            machine_profile=reader.string("machineProfile"),
            gpu_product=reader.string("gpuProduct"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"machineProfile": self.machine_profile, "gpuProduct": self.gpu_product}


@dataclass(frozen=True, slots=True, kw_only=True)
class ContainerShape:
    gpu_count: int
    cpu_cores: float
    memory_mib: int

    def __post_init__(self) -> None:
        _require_at_least("gpuCount", self.gpu_count, 1)
        _require_positive("cpuCores", self.cpu_cores)
        _require_at_least("memoryMiB", self.memory_mib, 1)

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(
            cls,
            gpu_count=reader.integer("gpuCount"),
            cpu_cores=reader.number("cpuCores"),
            memory_mib=reader.integer("memoryMiB"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"gpuCount": self.gpu_count, "cpuCores": self.cpu_cores, "memoryMiB": self.memory_mib}


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestShape:
    operation: Operation
    items_per_request: int
    input_tokens_per_item: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        _require_at_least("itemsPerRequest", self.items_per_request, 1)
        _require_at_least("inputTokensPerItem", self.input_tokens_per_item, 0)
        if self.operation is Operation.GENERATE:
            if self.output_tokens is None:
                raise SchemaError("outputTokens", "is required for the generate operation")
            _require_at_least("outputTokens", self.output_tokens, 1)
        else:
            _require_absent("outputTokens", self.output_tokens, "for an operation other than generate")

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(
            cls,
            operation=reader.enum("operation", Operation),
            items_per_request=reader.integer("itemsPerRequest"),
            input_tokens_per_item=reader.optional_integer("inputTokensPerItem"),
            output_tokens=reader.optional_integer("outputTokens"),
        )

    def to_dict(self) -> dict[str, object]:
        return _compact(
            {
                "operation": self.operation.value,
                "itemsPerRequest": self.items_per_request,
                "inputTokensPerItem": self.input_tokens_per_item,
                "outputTokens": self.output_tokens,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Throughput:
    requests_per_second: float
    output_tokens_per_second: float | None = None

    def __post_init__(self) -> None:
        _require_positive("requestsPerSecond", self.requests_per_second)
        _require_positive("outputTokensPerSecond", self.output_tokens_per_second)

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(
            cls,
            requests_per_second=reader.number("requestsPerSecond"),
            output_tokens_per_second=reader.optional_number("outputTokensPerSecond"),
        )

    def to_dict(self) -> dict[str, object]:
        return _compact(
            {"requestsPerSecond": self.requests_per_second, "outputTokensPerSecond": self.output_tokens_per_second}
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Percentiles:
    p99_ms: float
    p50_ms: float | None = None
    p95_ms: float | None = None

    def __post_init__(self) -> None:
        _require_at_least("p50Ms", self.p50_ms, 0)
        _require_at_least("p95Ms", self.p95_ms, 0)
        _require_at_least("p99Ms", self.p99_ms, 0)
        _require_ordered("p50Ms", self.p50_ms, "p95Ms", self.p95_ms)
        _require_ordered("p95Ms", self.p95_ms, "p99Ms", self.p99_ms)
        _require_ordered("p50Ms", self.p50_ms, "p99Ms", self.p99_ms)

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(
            cls,
            p50_ms=reader.optional_number("p50Ms"),
            p95_ms=reader.optional_number("p95Ms"),
            p99_ms=reader.number("p99Ms"),
        )

    def to_dict(self) -> dict[str, object]:
        return _compact({"p50Ms": self.p50_ms, "p95Ms": self.p95_ms, "p99Ms": self.p99_ms})


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceRef:
    uri: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        _require_identifier("uri", self.uri)
        if self.sha256 is not None and _SHA256.fullmatch(self.sha256) is None:
            raise SchemaError("sha256", f"must be 64 lowercase hexadecimal characters, got {self.sha256!r}")

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(cls, uri=reader.string("uri"), sha256=reader.optional_string("sha256"))

    def to_dict(self) -> dict[str, object]:
        return _compact({"uri": self.uri, "sha256": self.sha256})


@dataclass(frozen=True, slots=True, kw_only=True)
class Receipt:
    identity: str
    cell: Cell
    container_shape: ContainerShape
    concurrency: int
    request_shape: RequestShape
    throughput: Throughput
    source: SourceRef
    latency: Percentiles | None = None
    ttft: Percentiles | None = None
    itl: Percentiles | None = None

    def __post_init__(self) -> None:
        _require_identifier("identity", self.identity)
        _require_at_least("concurrency", self.concurrency, 1)
        if self.request_shape.operation is Operation.GENERATE:
            _require_absent("latency", self.latency, "for the generate operation, which reports ttft and itl")
            if self.ttft is None:
                raise SchemaError("ttft", "is required for the generate operation")
            if self.itl is None:
                raise SchemaError("itl", "is required for the generate operation")
        else:
            if self.latency is None:
                raise SchemaError("latency", "is required for an operation other than generate")
            _require_absent("ttft", self.ttft, "for an operation other than generate")
            _require_absent("itl", self.itl, "for an operation other than generate")
            _require_absent(
                "throughput.outputTokensPerSecond",
                self.throughput.output_tokens_per_second,
                "for an operation other than generate",
            )

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        reader.header(RECEIPT_KIND)
        return reader.build(
            cls,
            identity=reader.string("identity"),
            cell=reader.child("cell", Cell.from_dict),
            container_shape=reader.child("containerShape", ContainerShape.from_dict),
            concurrency=reader.integer("concurrency"),
            request_shape=reader.child("requestShape", RequestShape.from_dict),
            throughput=reader.child("throughput", Throughput.from_dict),
            latency=reader.optional_child("latency", Percentiles.from_dict),
            ttft=reader.optional_child("ttft", Percentiles.from_dict),
            itl=reader.optional_child("itl", Percentiles.from_dict),
            source=reader.child("source", SourceRef.from_dict),
        )

    def to_dict(self) -> dict[str, object]:
        return _compact(
            {
                "kind": RECEIPT_KIND,
                "schemaVersion": SCHEMA_VERSION,
                "identity": self.identity,
                "cell": self.cell.to_dict(),
                "containerShape": self.container_shape.to_dict(),
                "concurrency": self.concurrency,
                "requestShape": self.request_shape.to_dict(),
                "throughput": self.throughput.to_dict(),
                "latency": None if self.latency is None else self.latency.to_dict(),
                "ttft": None if self.ttft is None else self.ttft.to_dict(),
                "itl": None if self.itl is None else self.itl.to_dict(),
                "source": self.source.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class BootEvidence:
    identity: str
    cell: Cell
    container_shape: ContainerShape
    warmth: Warmth
    boot_seconds: float
    outcome: BootOutcome
    source: SourceRef

    def __post_init__(self) -> None:
        _require_identifier("identity", self.identity)
        if self.warmth not in BOOT_WARMTHS:
            choices = ", ".join(repr(warmth.value) for warmth in Warmth if warmth in BOOT_WARMTHS)
            raise SchemaError("warmth", f"boot evidence records one of {choices}, got {self.warmth.value!r}")
        _require_at_least("bootSeconds", self.boot_seconds, 0)

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        reader.header(BOOT_EVIDENCE_KIND)
        return reader.build(
            cls,
            identity=reader.string("identity"),
            cell=reader.child("cell", Cell.from_dict),
            container_shape=reader.child("containerShape", ContainerShape.from_dict),
            warmth=reader.enum("warmth", Warmth),
            boot_seconds=reader.number("bootSeconds"),
            outcome=reader.enum("outcome", BootOutcome),
            source=reader.child("source", SourceRef.from_dict),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": BOOT_EVIDENCE_KIND,
            "schemaVersion": SCHEMA_VERSION,
            "identity": self.identity,
            "cell": self.cell.to_dict(),
            "containerShape": self.container_shape.to_dict(),
            "warmth": self.warmth.value,
            "bootSeconds": self.boot_seconds,
            "outcome": self.outcome.value,
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class MachineProfilePrice:
    name: str
    gpu_product: str
    accelerator_per_hour: float
    core_per_hour: float
    memory_gib_per_hour: float

    def __post_init__(self) -> None:
        _require_identifier("name", self.name)
        _require_identifier("gpuProduct", self.gpu_product)
        _require_at_least("acceleratorPerHour", self.accelerator_per_hour, 0)
        _require_at_least("corePerHour", self.core_per_hour, 0)
        _require_at_least("memoryGiBPerHour", self.memory_gib_per_hour, 0)

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(
            cls,
            name=reader.string("name"),
            gpu_product=reader.string("gpuProduct"),
            accelerator_per_hour=reader.number("acceleratorPerHour"),
            core_per_hour=reader.number("corePerHour"),
            memory_gib_per_hour=reader.number("memoryGiBPerHour"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "gpuProduct": self.gpu_product,
            "acceleratorPerHour": self.accelerator_per_hour,
            "corePerHour": self.core_per_hour,
            "memoryGiBPerHour": self.memory_gib_per_hour,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Placement:
    name: str
    region_multiplier: float

    def __post_init__(self) -> None:
        _require_identifier("name", self.name)
        _require_positive("regionMultiplier", self.region_multiplier)

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(cls, name=reader.string("name"), region_multiplier=reader.number("regionMultiplier"))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "regionMultiplier": self.region_multiplier}


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceList:
    currency: str
    machine_profiles: tuple[MachineProfilePrice, ...]
    placements: tuple[Placement, ...]

    def __post_init__(self) -> None:
        if _CURRENCY.fullmatch(self.currency) is None:
            raise SchemaError("currency", f"must be a three-letter upper-case currency code, got {self.currency!r}")
        if not self.machine_profiles:
            raise SchemaError("machineProfiles", "must list at least one machine profile")
        if not self.placements:
            raise SchemaError("placements", "must list at least one placement")
        _require_unique("machineProfiles", "name", (profile.name for profile in self.machine_profiles))
        _require_unique("placements", "name", (placement.name for placement in self.placements))

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        reader.header(PRICE_LIST_KIND)
        return reader.build(
            cls,
            currency=reader.string("currency"),
            machine_profiles=reader.children("machineProfiles", MachineProfilePrice.from_dict),
            placements=reader.children("placements", Placement.from_dict),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": PRICE_LIST_KIND,
            "schemaVersion": SCHEMA_VERSION,
            "currency": self.currency,
            "machineProfiles": [profile.to_dict() for profile in self.machine_profiles],
            "placements": [placement.to_dict() for placement in self.placements],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class DemandHour:
    start: datetime
    requests: int
    units: int | None = None

    def __post_init__(self) -> None:
        _require_hour("start", self.start)
        _require_at_least("requests", self.requests, 0)
        _require_at_least("units", self.units, 0)

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(
            cls,
            start=reader.hour("start"),
            requests=reader.integer("requests"),
            units=reader.optional_integer("units"),
        )

    def to_dict(self) -> dict[str, object]:
        return _compact({"start": _format_hour(self.start), "requests": self.requests, "units": self.units})


@dataclass(frozen=True, slots=True, kw_only=True)
class DemandSeries:
    identity: str
    operation: Operation
    hours: tuple[DemandHour, ...]

    def __post_init__(self) -> None:
        _require_identifier("identity", self.identity)
        for index in range(1, len(self.hours)):
            if self.hours[index].start <= self.hours[index - 1].start:
                raise SchemaError(f"hours[{index}].start", "must be later than the previous hour")

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(
            cls,
            identity=reader.string("identity"),
            operation=reader.enum("operation", Operation),
            hours=reader.children("hours", DemandHour.from_dict),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "operation": self.operation.value,
            "hours": [hour.to_dict() for hour in self.hours],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Demand:
    start: datetime
    end: datetime
    series: tuple[DemandSeries, ...]

    def __post_init__(self) -> None:
        _require_hour("start", self.start)
        _require_hour("end", self.end)
        if self.end <= self.start:
            raise SchemaError("end", "must be later than start")
        _require_unique("series", "identity", (f"{entry.identity} {entry.operation.value}" for entry in self.series))
        for series_index, entry in enumerate(self.series):
            for hour_index, hour in enumerate(entry.hours):
                if not self.start <= hour.start < self.end:
                    raise SchemaError(
                        f"series[{series_index}].hours[{hour_index}].start",
                        f"must fall inside the window [{_format_hour(self.start)}, {_format_hour(self.end)})",
                    )

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        reader.header(DEMAND_KIND)
        return reader.build(
            cls,
            start=reader.hour("start"),
            end=reader.hour("end"),
            series=reader.children("series", DemandSeries.from_dict),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": DEMAND_KIND,
            "schemaVersion": SCHEMA_VERSION,
            "start": _format_hour(self.start),
            "end": _format_hour(self.end),
            "series": [entry.to_dict() for entry in self.series],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Envelope:
    floor_min: int
    floor_max: int
    buffer_min: int | None = None
    buffer_max: int | None = None
    scaledown_window_min_s: int | None = None
    scaledown_window_max_s: int | None = None

    def __post_init__(self) -> None:
        for name, value in self._fields():
            _require_at_least(name, value, 0)
        _require_ordered("floorMin", self.floor_min, "floorMax", self.floor_max)
        _require_pair("bufferMin", self.buffer_min, "bufferMax", self.buffer_max)
        _require_ordered("bufferMin", self.buffer_min, "bufferMax", self.buffer_max)
        _require_pair(
            "scaledownWindowMinS", self.scaledown_window_min_s, "scaledownWindowMaxS", self.scaledown_window_max_s
        )
        _require_ordered(
            "scaledownWindowMinS", self.scaledown_window_min_s, "scaledownWindowMaxS", self.scaledown_window_max_s
        )

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(
            cls,
            floor_min=reader.integer("floorMin"),
            floor_max=reader.integer("floorMax"),
            buffer_min=reader.optional_integer("bufferMin"),
            buffer_max=reader.optional_integer("bufferMax"),
            scaledown_window_min_s=reader.optional_integer("scaledownWindowMinS"),
            scaledown_window_max_s=reader.optional_integer("scaledownWindowMaxS"),
        )

    def _fields(self) -> tuple[tuple[str, int | None], ...]:
        return (
            ("floorMin", self.floor_min),
            ("floorMax", self.floor_max),
            ("bufferMin", self.buffer_min),
            ("bufferMax", self.buffer_max),
            ("scaledownWindowMinS", self.scaledown_window_min_s),
            ("scaledownWindowMaxS", self.scaledown_window_max_s),
        )

    def to_dict(self) -> dict[str, object]:
        return _compact(dict(self._fields()))


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanLane:
    name: str
    queue_pool: str
    bundle: str
    cell: Cell
    container_shape: ContainerShape
    concurrency: int
    max_containers: int
    envelope: Envelope

    def __post_init__(self) -> None:
        _require_identifier("name", self.name)
        _require_identifier("queuePool", self.queue_pool)
        _require_identifier("bundle", self.bundle)
        _require_at_least("concurrency", self.concurrency, 1)
        _require_at_least("maxContainers", self.max_containers, 1)
        if self.envelope.floor_max > self.max_containers:
            raise SchemaError(
                "envelope.floorMax",
                f"must not exceed maxContainers ({self.max_containers}), got {self.envelope.floor_max}",
            )

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(
            cls,
            name=reader.string("name"),
            queue_pool=reader.string("queuePool"),
            bundle=reader.string("bundle"),
            cell=reader.child("cell", Cell.from_dict),
            container_shape=reader.child("containerShape", ContainerShape.from_dict),
            concurrency=reader.integer("concurrency"),
            max_containers=reader.integer("maxContainers"),
            envelope=reader.child("envelope", Envelope.from_dict),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "queuePool": self.queue_pool,
            "bundle": self.bundle,
            "cell": self.cell.to_dict(),
            "containerShape": self.container_shape.to_dict(),
            "concurrency": self.concurrency,
            "maxContainers": self.max_containers,
            "envelope": self.envelope.to_dict(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanIdentity:
    identity: str
    lane: str
    warmth: Warmth

    def __post_init__(self) -> None:
        _require_identifier("identity", self.identity)
        _require_identifier("lane", self.lane)

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        return reader.build(
            cls,
            identity=reader.string("identity"),
            lane=reader.string("lane"),
            warmth=reader.enum("warmth", Warmth),
        )

    def to_dict(self) -> dict[str, object]:
        return {"identity": self.identity, "lane": self.lane, "warmth": self.warmth.value}


@dataclass(frozen=True, slots=True, kw_only=True)
class Plan:
    placement: str
    lanes: tuple[PlanLane, ...]
    identities: tuple[PlanIdentity, ...]

    def __post_init__(self) -> None:
        _require_identifier("placement", self.placement)
        _require_unique("lanes", "name", (lane.name for lane in self.lanes))
        _require_unique("identities", "identity", (entry.identity for entry in self.identities))
        lanes = {lane.name: lane for lane in self.lanes}
        for index, entry in enumerate(self.identities):
            lane = lanes.get(entry.lane)
            if lane is None:
                raise SchemaError(f"identities[{index}].lane", f"names no lane of this plan: {entry.lane!r}")
            warm_floor = lane.envelope.floor_min >= 1
            if (entry.warmth in WARM_FLOOR_WARMTHS) != warm_floor:
                expectation = "at least 1" if entry.warmth in WARM_FLOOR_WARMTHS else "0"
                raise SchemaError(
                    f"identities[{index}].warmth",
                    f"{entry.warmth.value} needs lane {lane.name!r} to have floorMin {expectation}, "
                    f"got {lane.envelope.floor_min}",
                )
        hosted = {entry.lane for entry in self.identities}
        for index, lane in enumerate(self.lanes):
            if lane.name not in hosted:
                raise SchemaError(f"lanes[{index}].name", f"lane {lane.name!r} hosts no identity")

    @classmethod
    def from_dict(cls, data: object, path: str = ROOT_PATH) -> Self:
        reader = _Reader(data, path)
        reader.header(PLAN_KIND)
        return reader.build(
            cls,
            placement=reader.string("placement"),
            lanes=reader.children("lanes", PlanLane.from_dict),
            identities=reader.children("identities", PlanIdentity.from_dict),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": PLAN_KIND,
            "schemaVersion": SCHEMA_VERSION,
            "placement": self.placement,
            "lanes": [lane.to_dict() for lane in self.lanes],
            "identities": [entry.to_dict() for entry in self.identities],
        }
