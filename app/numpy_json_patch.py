from __future__ import annotations

import json
from typing import Any

import numpy as np


_ORIGINAL_DUMPS = json.dumps


def _numpy_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _safe_dumps(obj: Any, *args: Any, **kwargs: Any) -> str:
    kwargs.setdefault("default", _numpy_default)
    return _ORIGINAL_DUMPS(obj, *args, **kwargs)


if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]


if not getattr(json.dumps, "_stemforge_numpy_safe", False):
    setattr(_safe_dumps, "_stemforge_numpy_safe", True)
    json.dumps = _safe_dumps
