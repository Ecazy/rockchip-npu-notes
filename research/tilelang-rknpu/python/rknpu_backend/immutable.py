# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only container types for data that must not change after it is built.

The execution plan is the single authoritative representation the runtime follows,
so two different problems have to be solved:

* **aliasing** -- a caller keeps a reference to the mapping passed into a node and
  edits it later.  A construction-time deep copy handles this.
* **post-construction mutation** -- a caller edits the mapping through a public
  accessor such as ``artifact.plan``.  ``frozen=True`` on a dataclass does not stop
  ``action.attrs["flag"] = 1``, and a descriptor is encoded from exactly those
  attributes, so such an edit could change execution while ``plan_hash`` (computed
  at construction) still reports the old plan.

:class:`FrozenDict` and :class:`FrozenList` are drop-in read-only subclasses of
``dict``/``list``: they keep ``isinstance(x, dict)`` and equality with plain
mappings working (both are used by the schema decoders and the tests), they are
JSON-serializable, and every mutating method raises ``TypeError``/``AttributeError``
instead of silently rewriting a compiled plan.

``freeze`` is applied where a record becomes authoritative; ``thaw`` produces plain
containers for serialization, so the on-disk JSON and the canonical plan hash are
unchanged by the read-only representation.
"""
from __future__ import annotations

from typing import Any

__all__ = ["FrozenDict", "FrozenList", "freeze", "thaw", "is_frozen"]

_MUTATION_ERROR = ("this mapping is part of an immutable compiled plan; "
                   "build a new value instead of mutating it")
_SEQUENCE_ERROR = ("this sequence is part of an immutable compiled plan; "
                   "build a new value instead of mutating it")


def _refuse(*_args: Any, **_kwargs: Any) -> None:
    raise TypeError(_MUTATION_ERROR)


def _refuse_sequence(*_args: Any, **_kwargs: Any) -> None:
    raise TypeError(_SEQUENCE_ERROR)


class FrozenDict(dict):
    """A ``dict`` that refuses mutation but still looks and compares like a dict."""

    __slots__ = ()

    __setitem__ = _refuse
    __delitem__ = _refuse
    __ior__ = _refuse
    clear = _refuse
    pop = _refuse
    popitem = _refuse
    setdefault = _refuse
    update = _refuse

    def __copy__(self) -> "FrozenDict":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrozenDict":
        # Sharing is safe because the object (and, after ``freeze``, everything it
        # holds) can never change.
        return self

    def __reduce__(self):
        return (FrozenDict, (dict(self),))


class FrozenList(list):
    """A ``list`` that refuses mutation but still looks and compares like a list."""

    __slots__ = ()

    __setitem__ = _refuse_sequence
    __delitem__ = _refuse_sequence
    __iadd__ = _refuse_sequence
    __imul__ = _refuse_sequence
    append = _refuse_sequence
    clear = _refuse_sequence
    extend = _refuse_sequence
    insert = _refuse_sequence
    pop = _refuse_sequence
    remove = _refuse_sequence
    reverse = _refuse_sequence
    sort = _refuse_sequence

    def __copy__(self) -> "FrozenList":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrozenList":
        return self

    def __reduce__(self):
        return (FrozenList, (list(self),))


def freeze(value: Any) -> Any:
    """Return a deep, read-only copy of ``value``.

    Mappings and sequences become :class:`FrozenDict`/:class:`FrozenList`; scalars
    are returned as-is.  The input is copied, so later edits to it cannot reach the
    frozen result.
    """
    if isinstance(value, FrozenDict) and all(
            is_frozen(item) for item in value.values()):
        return value
    if isinstance(value, FrozenList) and all(is_frozen(item) for item in value):
        return value
    if isinstance(value, dict):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return FrozenList([freeze(item) for item in value])
    return value


def thaw(value: Any) -> Any:
    """Return a plain ``dict``/``list`` projection of a frozen value."""
    if isinstance(value, dict):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(item) for item in value]
    return value


def is_frozen(value: Any) -> bool:
    """True when ``value`` carries no mutable container anywhere inside it."""
    if isinstance(value, FrozenDict):
        return all(is_frozen(item) for item in value.values())
    if isinstance(value, FrozenList):
        return all(is_frozen(item) for item in value)
    if isinstance(value, (dict, list, set, bytearray)):
        return False
    return True
