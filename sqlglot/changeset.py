"""
Serializable, replayable changesets for SQL expression trees.

While :func:`sqlglot.diff` returns edits that reference in-memory ``Expression``
objects, a *changeset* describes the same transformation using only
JSON-serializable data: nodes are addressed by structural paths (no Python object
addresses), guarded by content fingerprints, and inserted/updated subtrees are
stored as serialized expression payloads. A changeset can be persisted with
``json.dumps`` and later replayed on a freshly parsed copy of the source SQL -
even in another process - with :func:`apply_changeset`.
"""

from __future__ import annotations

import hashlib
import json
import typing as t
from itertools import chain

from sqlglot import expressions as exp
from sqlglot.diff import IGNORED_LEAF_EXPRESSION_TYPES, Keep, Move, Update, diff
from sqlglot.serde import dump as _dump_expression, load as _load_expression

__all__ = ["ChangesetConflict", "apply_changeset", "diff_changeset"]

CHANGESET_VERSION = 1

# A structural path addresses a node from the root of its tree: each step is an
# [arg_key, index] pair, where index is None for scalar args.
Path = list[list[t.Any]]

Changeset = dict[str, t.Any]


class ChangesetConflict(Exception):
    """
    Raised when a changeset is applied to a tree that no longer matches the
    source tree the changeset was computed from.

    Attributes:
        path: the structural path to the node where the conflict was detected.
            An empty path means the tree as a whole doesn't match the changeset's
            source fingerprint (e.g. it was modified in a region no edit touches).
        op_index: the index of the changeset edit that detected the conflict, if any.
    """

    def __init__(self, path: Path, op_index: int | None = None) -> None:
        self.path = path
        self.op_index = op_index

        location = (
            "root"
            if not path
            else " -> ".join(
                f"{key}[{index}]" if index is not None else str(key) for key, index in path
            )
        )
        message = f"Changeset conflict at {location}"
        if op_index is not None:
            message = f"{message} (edit #{op_index})"
        super().__init__(message)


def diff_changeset(
    source: exp.Expr,
    target: exp.Expr,
    matchings: list[tuple[exp.Expr, exp.Expr]] | None = None,
    **kwargs: t.Any,
) -> Changeset:
    """
    Computes a JSON-serializable changeset that transforms ``source`` into ``target``.

    Args:
        source: the source expression.
        target: the target expression against which the diff is calculated.
        matchings: optional list of pre-matched node pairs, forwarded to ``sqlglot.diff``.
        kwargs: additional arguments forwarded to ``sqlglot.diff``.

    Returns:
        A JSON-serializable dict with the edits needed to turn ``source`` into
        ``target``. Applying it with :func:`apply_changeset` to a tree equal to
        ``source`` produces a tree equal to ``target``.
    """
    work = _prepare(source)
    goal = _prepare(target)

    if matchings:
        source_map = {id(old): new for old, new in zip(source.walk(), work.walk())}
        target_map = {id(old): new for old, new in zip(target.walk(), goal.walk())}
        matchings = [(source_map[id(s)], target_map[id(t)]) for s, t in matchings]

    kwargs.pop("delta_only", None)
    edits = diff(work, goal, matchings=matchings, delta_only=False, **kwargs)
    return _ChangesetBuilder(work, goal, edits).build()


def apply_changeset(tree: exp.Expr, changeset: Changeset | str | bytes) -> exp.Expr:
    """
    Replays a changeset produced by :func:`diff_changeset` on ``tree``.

    The changeset is bound to the structure and content of the source tree it was
    computed from: before each edit is applied, the affected node is checked
    against the fingerprint recorded in the changeset. The edits are applied to a
    copy, so the group either applies successfully as a whole or ``tree`` is left
    untouched.

    Args:
        tree: the expression the changeset should be applied to. Never mutated.
        changeset: a changeset dict, or its JSON serialization.

    Returns:
        A new expression equal to the changeset's target tree.

    Raises:
        ChangesetConflict: if ``tree`` no longer matches the changeset's source tree.
    """
    if isinstance(changeset, (str, bytes, bytearray)):
        changeset = t.cast(Changeset, json.loads(changeset))
    if changeset.get("version") != CHANGESET_VERSION:
        raise ValueError(f"Unsupported changeset version: {changeset.get('version')}")

    work = tree.copy()
    matches_source = _fingerprint(work) == changeset["fingerprint"]
    for op_index, op in enumerate(changeset["edits"]):
        work = _apply_op(work, op, op_index)
    if not matches_source:
        raise ChangesetConflict([])
    return work


def _apply_op(work: exp.Expr, op: dict[str, t.Any], op_index: int) -> exp.Expr:
    kind = op["op"]
    node = _resolve(work, op["path"], op_index)

    if kind in ("remove", "replace", "move"):
        expected = _fingerprint(node)
    elif kind == "update":
        expected = _shallow_fingerprint(node)
    elif kind == "insert":
        expected = (
            _list_fingerprint(node, op["arg"])
            if op["index"] is not None
            else _shallow_fingerprint(node)
        )
    else:
        raise ValueError(f"Unknown changeset op: {kind}")

    if expected != op["expected"]:
        raise ChangesetConflict(op["path"], op_index)

    if kind == "remove":
        node.pop()
    elif kind == "update":
        for key in op["shallow"]:
            node.set(key, None)
        for key, value in _load(op["args"]).args.items():
            node.set(key, value)
    elif kind == "replace":
        replacement = _load(op["node"])
        if node.parent is None:
            return replacement
        node.replace(replacement)
    elif kind == "insert":
        child = _load(op["node"])
        if op["index"] is None:
            if node.args.get(op["arg"]) is not None:
                raise ChangesetConflict(op["path"], op_index)
            node.set(op["arg"], child)
        else:
            _insert_at(node, op["arg"], op["index"], child)
    elif kind == "move":
        parent = _resolve(work, op["parent"], op_index)
        if _list_fingerprint(parent, op["arg"]) != op["parent_expected"]:
            raise ChangesetConflict(op["parent"], op_index)
        node.pop()
        _insert_at(parent, op["arg"], op["index"], node)

    return work


class _ChangesetBuilder:
    """
    Turns the edit script produced by ``sqlglot.diff`` into an executable,
    serializable changeset.

    Edits are emitted against a simulated working copy of the source tree, so
    every recorded path, list index and fingerprint is valid at the exact moment
    the edit is replayed. This makes the changeset robust to list index shifts
    caused by earlier moves, inserts and removes.
    """

    def __init__(self, work: exp.Expr, goal: exp.Expr, edits: list[t.Any]) -> None:
        self.work = work
        self.goal = goal
        self.ops: list[dict[str, t.Any]] = []
        self.fingerprint = _fingerprint(work)

        work_by_id = {id(node): node for node in work.walk()}
        matched = {}
        for edit in edits:
            if isinstance(edit, (Keep, Move, Update)) and id(edit.source) in work_by_id:
                matched[id(edit.source)] = edit.target

        self.matched = _filter_to_consistent(matched, work_by_id)
        self.goal_to_work = {id(t): work_by_id[s] for s, t in self.matched.items()}

    def build(self) -> Changeset:
        if id(self.work) not in self.matched:
            self.ops.append(
                {
                    "op": "replace",
                    "path": [],
                    "expected": self.fingerprint,
                    "node": _dump_expression(self.goal),
                }
            )
        else:
            self._sync(self.work, self.goal)

        return {"version": CHANGESET_VERSION, "fingerprint": self.fingerprint, "edits": self.ops}

    def _sync(self, source: exp.Expr, target: exp.Expr) -> None:
        keys = list(dict.fromkeys(chain(source.args, target.args)))
        shallow = [
            key
            for key in keys
            if _is_shallow_value(source.args.get(key)) or _is_shallow_value(target.args.get(key))
        ]

        if any(source.args.get(key) != target.args.get(key) for key in shallow):
            shell = type(target)()
            shell.args = {key: target.args[key] for key in shallow if key in target.args}
            payload = _dump_expression(shell)
            self.ops.append(
                {
                    "op": "update",
                    "path": _path(source),
                    "expected": _shallow_fingerprint(source),
                    "shallow": shallow,
                    "args": payload,
                }
            )
            loaded = _load(payload)
            for key in shallow:
                source.set(key, None)
            for key, value in loaded.args.items():
                source.set(key, value)

        for key in keys:
            if key in shallow:
                continue
            source_value = source.args.get(key)
            target_value = target.args.get(key)

            if isinstance(source_value, list) or isinstance(target_value, list):
                self._sync_list(source, target, key)
            elif target_value is None:
                self._remove(t.cast(exp.Expr, source_value))
            elif source_value is None:
                payload = _dump_expression(target_value)
                self.ops.append(
                    {
                        "op": "insert",
                        "path": _path(source),
                        "expected": _shallow_fingerprint(source),
                        "arg": key,
                        "index": None,
                        "node": payload,
                    }
                )
                source.set(key, _load(payload))
            elif self.matched.get(id(source_value)) is target_value:
                self._sync(source_value, t.cast(exp.Expr, target_value))
            else:
                payload = _dump_expression(target_value)
                self.ops.append(
                    {
                        "op": "replace",
                        "path": _path(source_value),
                        "expected": _fingerprint(source_value),
                        "node": payload,
                    }
                )
                source_value.replace(_load(payload))

    def _sync_list(self, source: exp.Expr, target: exp.Expr, key: str) -> None:
        source_value = source.args.get(key)
        target_value = target.args.get(key)
        source_items = source_value if isinstance(source_value, list) else []
        target_items = target_value if isinstance(target_value, list) else []

        desired: list[exp.Expr | None] = []
        for item in target_items:
            match = self.goal_to_work.get(id(item))
            if match is not None and match.parent is source and match.arg_key == key:
                desired.append(match)
            else:
                desired.append(None)

        keep = {id(item) for item in desired if item is not None}
        for item in list(source_items):
            if id(item) not in keep:
                self._remove(item)

        for index, (item, target_item) in enumerate(zip(desired, target_items)):
            if item is None:
                payload = _dump_expression(target_item)
                self.ops.append(
                    {
                        "op": "insert",
                        "path": _path(source),
                        "expected": _list_fingerprint(source, key),
                        "arg": key,
                        "index": index,
                        "node": payload,
                    }
                )
                _insert_at(source, key, index, _load(payload))
            else:
                current = _index_of(source.args[key], item)
                if current != index:
                    self.ops.append(
                        {
                            "op": "move",
                            "path": _path(item),
                            "expected": _fingerprint(item),
                            "parent": _path(source),
                            "parent_expected": _list_fingerprint(source, key),
                            "arg": key,
                            "index": index,
                        }
                    )
                    item.pop()
                    _insert_at(source, key, index, item)

        for item, target_item in zip(desired, target_items):
            if item is not None:
                self._sync(item, target_item)

    def _remove(self, node: exp.Expr) -> None:
        self.ops.append({"op": "remove", "path": _path(node), "expected": _fingerprint(node)})
        node.pop()


def _prepare(tree: exp.Expr) -> exp.Expr:
    copy = tree.copy()
    ids = [id(node) for node in copy.walk()]
    if len(set(ids)) != len(ids):
        # The tree shares nodes between parents, which ``diff`` cannot handle
        # directly. A serde round-trip rebuilds it with unique nodes.
        copy = _load(_dump_expression(tree))
    return copy


def _load(payloads: list[dict[str, t.Any]]) -> exp.Expr:
    return t.cast(exp.Expr, _load_expression(payloads))


def _filter_to_consistent(
    matched: dict[int, exp.Expr], work_by_id: dict[int, exp.Expr]
) -> dict[int, exp.Expr]:
    """
    Keeps only matchings whose parents are also matched to each other, so that
    every remaining pair can be synced in place. Nodes dropped here (e.g. ones
    that were re-parented) are handled as removes + inserts instead.
    """

    def parents_consistent(source_id: int, target: exp.Expr) -> bool:
        source_parent = work_by_id[source_id].parent
        target_parent = target.parent
        if source_parent is None or target_parent is None:
            return source_parent is None and target_parent is None
        return matched.get(id(source_parent)) is target_parent

    matched = {s: t for s, t in matched.items() if s in work_by_id}
    while True:
        inconsistent = [s for s, t in matched.items() if not parents_consistent(s, t)]
        if not inconsistent:
            return matched
        for source_id in inconsistent:
            del matched[source_id]


def _is_shallow_value(value: t.Any) -> bool:
    if isinstance(value, exp.Expr):
        return isinstance(value, IGNORED_LEAF_EXPRESSION_TYPES)
    if isinstance(value, list):
        return all(_is_shallow_value(item) for item in value)
    return True


def _insert_at(parent: exp.Expr, key: str, index: int, node: exp.Expr) -> None:
    values = parent.args.get(key)
    if not isinstance(values, list) or index >= len(values):
        parent.append(key, node)
    else:
        parent.set(key, node, index=index, overwrite=False)


def _index_of(items: list[t.Any], node: exp.Expr) -> int:
    for index, item in enumerate(items):
        if item is node:
            return index
    return -1


def _path(node: exp.Expr) -> Path:
    steps = []
    while node.parent is not None:
        parent = node.parent
        for key, value in parent.args.items():
            if isinstance(value, list):
                index = _index_of(value, node)
                if index >= 0:
                    steps.append([key, index])
                    break
            elif value is node:
                steps.append([key, None])
                break
        else:
            raise ValueError("Cannot compute the path of a detached node")
        node = parent
    steps.reverse()
    return steps


def _resolve(root: exp.Expr, path: Path, op_index: int | None = None) -> exp.Expr:
    node: t.Any = root
    for key, index in path:
        value = node.args.get(key) if isinstance(node, exp.Expr) else None
        if index is None:
            node = value
        elif isinstance(value, list) and 0 <= index < len(value):
            node = value[index]
        else:
            node = None
        if not isinstance(node, exp.Expr):
            raise ChangesetConflict(path, op_index)
    return node


def _fingerprint(node: exp.Expr) -> str:
    canonical = json.dumps(_canonical(node), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _shallow_fingerprint(node: exp.Expr) -> str:
    shallow = [
        [key, _canonical_value(value)]
        for key, value in sorted(node.args.items())
        if _is_shallow_value(value)
    ]
    canonical = json.dumps(
        [node.__class__.__qualname__, node.comments or [], shallow], separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _list_fingerprint(node: exp.Expr, key: str) -> str:
    values = node.args.get(key) or []
    canonical = json.dumps([_canonical_value(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _canonical(node: exp.Expr) -> list[t.Any]:
    return [
        node.__class__.__qualname__,
        node.comments or [],
        [[key, _canonical_value(value)] for key, value in sorted(node.args.items())],
    ]


def _canonical_value(value: t.Any) -> t.Any:
    if isinstance(value, exp.Expr):
        return _canonical(value)
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
