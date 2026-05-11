from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel


def strip_extra_fields(model_type: type[BaseModel], data: Any) -> Any:
    if not isinstance(data, dict):
        return data

    cleaned: dict[str, Any] = {}
    for field_name, field_info in model_type.model_fields.items():
        if field_name not in data:
            continue
        cleaned[field_name] = _strip_value(field_info.annotation, data[field_name])
    return cleaned


def _strip_value(annotation: Any, value: Any) -> Any:
    if value is None:
        return None

    if _is_model_type(annotation):
        return strip_extra_fields(annotation, value)

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list and isinstance(value, list):
        item_annotation = args[0] if args else Any
        return [_strip_value(item_annotation, item) for item in value]

    if origin in {Union, UnionType}:
        return _strip_union_value(args, value)

    return value


def _strip_union_value(args: tuple[Any, ...], value: Any) -> Any:
    for annotation in args:
        if annotation is type(None):
            continue
        if _is_model_type(annotation) and isinstance(value, dict):
            return strip_extra_fields(annotation, value)
        origin = get_origin(annotation)
        if origin is list and isinstance(value, list):
            item_annotation = get_args(annotation)[0] if get_args(annotation) else Any
            return [_strip_value(item_annotation, item) for item in value]
    return value


def _is_model_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)
