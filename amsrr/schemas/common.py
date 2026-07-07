from __future__ import annotations

import json
import types
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar, get_args, get_origin, get_type_hints


class SchemaValidationError(ValueError):
    """Raised when a schema object violates the v0.4 contract."""


Pose7D = tuple[float, float, float, float, float, float, float]
Vector3 = tuple[float, float, float]
Condition = dict[str, Any]

T = TypeVar("T", bound="SchemaBase")


class StrEnum(str, Enum):
    """String enum with a small helper for validation messages."""

    @classmethod
    def values(cls) -> list[str]:
        return [item.value for item in cls]


class ContactMode(StrEnum):
    GRASP = "grasp"
    SUPPORT = "support"
    PUSH = "push"
    LATCH = "latch"
    PERCH = "perch"
    SLIDE = "slide"
    STICK = "stick"
    FREE_FLIGHT = "free_flight"
    BODY_CONTACT = "body_contact"
    TOOL = "tool"


def _field_path(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _is_union(origin: Any) -> bool:
    return origin is types.UnionType or str(origin) == "typing.Union"


def _coerce_primitive(value: Any, target_type: type, path: str) -> Any:
    if target_type is Any:
        return value
    if target_type is bool:
        if isinstance(value, bool):
            return value
        raise SchemaValidationError(f"{path} must be bool")
    if target_type is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lstrip("-").isdigit():
            return int(value)
        raise SchemaValidationError(f"{path} must be int")
    if target_type is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise SchemaValidationError(f"{path} must be float")
    if target_type is str:
        if isinstance(value, str):
            return value
        raise SchemaValidationError(f"{path} must be str")
    if target_type is Path:
        return Path(value)
    return value


def coerce_value(value: Any, target_type: Any, path: str = "") -> Any:
    """Coerce JSON/YAML-compatible data into a schema field type."""

    if target_type is Any:
        return value
    if target_type is None or target_type is type(None):
        if value is None:
            return None
        raise SchemaValidationError(f"{path} must be null")
    if target_type is slice:
        if isinstance(value, slice):
            return value
        if isinstance(value, dict):
            return slice(value.get("start"), value.get("stop"), value.get("step"))
        if isinstance(value, (list, tuple)) and len(value) in (2, 3):
            return slice(*value)
        raise SchemaValidationError(f"{path} must be a slice mapping or sequence")

    origin = get_origin(target_type)
    args = get_args(target_type)

    if origin is Literal:
        if value not in args:
            raise SchemaValidationError(f"{path} must be one of {list(args)}, got {value!r}")
        return value

    if _is_union(origin):
        errors: list[str] = []
        for option in args:
            try:
                return coerce_value(value, option, path)
            except SchemaValidationError as exc:
                errors.append(str(exc))
        raise SchemaValidationError(f"{path} did not match any allowed type: {errors}")

    if origin is list:
        if not isinstance(value, list):
            raise SchemaValidationError(f"{path} must be list")
        item_type = args[0] if args else Any
        return [coerce_value(item, item_type, f"{path}[{idx}]") for idx, item in enumerate(value)]

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise SchemaValidationError(f"{path} must be tuple/list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(coerce_value(item, args[0], f"{path}[{idx}]") for idx, item in enumerate(value))
        if args and len(value) != len(args):
            raise SchemaValidationError(f"{path} must have length {len(args)}, got {len(value)}")
        return tuple(
            coerce_value(item, args[idx] if args else Any, f"{path}[{idx}]")
            for idx, item in enumerate(value)
        )

    if origin is dict:
        if not isinstance(value, dict):
            raise SchemaValidationError(f"{path} must be dict")
        key_type = args[0] if args else Any
        value_type = args[1] if len(args) > 1 else Any
        return {
            coerce_value(key, key_type, f"{path}.key"): coerce_value(item, value_type, f"{path}[{key!r}]")
            for key, item in value.items()
        }

    if isinstance(target_type, type) and issubclass(target_type, Enum):
        if isinstance(value, target_type):
            return value
        try:
            return target_type(value)
        except ValueError as exc:
            raise SchemaValidationError(
                f"{path} must be one of {[item.value for item in target_type]}, got {value!r}"
            ) from exc

    if isinstance(target_type, type) and is_dataclass(target_type):
        if isinstance(value, target_type):
            return value
        if not isinstance(value, dict):
            raise SchemaValidationError(f"{path} must be object")
        if issubclass(target_type, SchemaBase):
            return target_type.from_dict(value)
        return target_type(**value)

    if isinstance(target_type, type):
        return _coerce_primitive(value, target_type, path)

    return value


def to_plain_data(value: Any) -> Any:
    """Convert schema objects into deterministic JSON-compatible data."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if is_dataclass(value):
        return {field.name: to_plain_data(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [to_plain_data(item) for item in value]
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(data: Any) -> str:
    return json.dumps(to_plain_data(data), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass
class SchemaBase:
    """Base class for strict dataclass schemas."""

    _allow_extra_fields: ClassVar[bool] = False

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        if not isinstance(data, dict):
            raise SchemaValidationError(f"{cls.__name__}.from_dict expects a dict")
        dataclass_fields = {field.name: field for field in fields(cls)}
        unknown = set(data) - set(dataclass_fields)
        if unknown and not cls._allow_extra_fields:
            raise SchemaValidationError(f"{cls.__name__} got unknown fields: {sorted(unknown)}")

        hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for name, field in dataclass_fields.items():
            if name not in data:
                if field.default is MISSING and field.default_factory is MISSING:
                    raise SchemaValidationError(f"{cls.__name__}.{name} is required")
                continue
            kwargs[name] = coerce_value(data[name], hints.get(name, Any), f"{cls.__name__}.{name}")
        return cls(**kwargs)

    @classmethod
    def from_json(cls: type[T], text: str) -> T:
        return cls.from_dict(json.loads(text))

    def to_dict(self) -> dict[str, Any]:
        return to_plain_data(self)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent, ensure_ascii=True)

    def stable_hash(self) -> str:
        from hashlib import sha256

        return sha256(canonical_json(self).encode("utf-8")).hexdigest()

    def validate(self) -> None:
        return None


def require_non_empty(value: str, path: str) -> None:
    if not value:
        raise SchemaValidationError(f"{path} must be non-empty")


def require_positive(value: float, path: str) -> None:
    if value <= 0:
        raise SchemaValidationError(f"{path} must be positive")


def require_non_negative(value: float, path: str) -> None:
    if value < 0:
        raise SchemaValidationError(f"{path} must be non-negative")


def require_len(value: list[Any] | tuple[Any, ...], expected: int, path: str) -> None:
    if len(value) != expected:
        raise SchemaValidationError(f"{path} must have length {expected}, got {len(value)}")
