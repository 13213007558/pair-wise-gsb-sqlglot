"""
Serializable, replayable changesets for expression trees.

This module builds on sqlglot.diff: instead of returning edits that reference in-memory
expression nodes, it produces a JSON-serializable changeset that can be stored and later
replayed on a re-parsed copy of the source tree.
"""

from __future__ import annotations

import hashlib
import json
import typing as t
from collections import defaultdict
from dataclasses import dataclass

from sqlglot import expressions as exp
from sqlglot import serde
from sqlglot.diff import Insert, Keep, Move, Remove, Update, diff

INSERT = "insert"
REMOVE = "remove"
UPDATE = "update"
MOVE = "move"

VERSION = 1

# A path addresses a node structurally: each step is an (arg_key, index) pair, where
# index is the position within a list arg (None for scalar args).
if t.TYPE_CHECKING:
    Path = tuple[tuple[str, int | None], ...]


@dataclass(frozen=True)
class Conflict:
    """A node in the tree being patched that doesn't match the changeset's expectations."""

    path: Path
    expected: str
    actual: str | None

    def __str__(self) -> str:
        return f"node at {_format_path(self.path)} (expected fingerprint {self.expected}, found {self.actual or 'nothing'})"


class ChangesetConflictError(Exception):
    """Raised when a changeset cannot be applied because the target tree has changed."""

    def __init__(self, conflicts: list[Conflict]) -> None:
        self.conflicts = conflicts
        details = "\n".join(f"  - {conflict}" for conflict in conflicts)
        super().__init__(f"Cannot apply changeset, {len(conflicts)} conflict(s) found:\n{details}")


@dataclass
class Changeset:
    """
    A serializable set of edits that transforms the source tree of a diff into its target tree.

    Nodes are addressed by their structural path within the source tree and guarded by content
    fingerprints, so applying a changeset to a tree that has been modified in the meantime fails
    with a ChangesetConflictError before anything is mutated.
    """

    edits: list[dict[str, t.Any]]

    def to_dict(self) -> dict[str, t.Any]:
        return {"version": VERSION, "edits": self.edits}

    @classmethod
    def from_dict(cls, payload: dict[str, t.Any]) -> Changeset:
        version = payload.get("version")
        if version != VERSION:
            raise ValueError(f"Unsupported changeset version: {version}")
        return cls(edits=payload["edits"])

    def to_json(self, **kwargs: t.Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> Changeset:
        return cls.from_dict(json.loads(payload))

    def apply(self, tree: exp.Expr) -> exp.Expr:
        return apply(tree, self)


def changeset(source: exp.Expr, target: exp.Expr, **kwargs: t.Any) -> Changeset:
    """
    Computes a serializable changeset that transforms source into target.

    Args:
        source: the source expression.
        target: the target expression against which the diff is calculated.
        kwargs: additional arguments to pass to the ChangeDistiller instance (e.g. f, t,
            dialect). Note that matchings is not supported, as the trees are copied.

    Returns:
        a Changeset that can be serialized with to_json and replayed with apply.
    """
    source = source.copy()
    target = target.copy()

    edits = diff(source, target, delta_only=False, **kwargs)
    source_paths = {id(node): path for path, node in _index_tree(source).items()}

    matched: dict[int, exp.Expr] = {}
    removed_ids: set[int] = set()
    inserted_ids: set[int] = set()
    updated_ids: set[int] = set()
    updated_target_ids: set[int] = set()
    move_candidates: dict[int, Move] = {}

    for edit in edits:
        if isinstance(edit, (Keep, Update, Move)):
            matched[id(edit.target)] = edit.source
        if isinstance(edit, Update):
            updated_ids.add(id(edit.source))
            updated_target_ids.add(id(edit.target))
        elif isinstance(edit, Remove):
            removed_ids.add(id(edit.expression))
        elif isinstance(edit, Insert):
            inserted_ids.add(id(edit.expression))
        elif isinstance(edit, Move):
            move_candidates.setdefault(id(edit.source), edit)

    # If the roots don't match, the whole tree is replaced and no other edit is needed
    if id(source) in removed_ids and id(target) in inserted_ids:
        return Changeset(edits=[_update_edit((), source, target)])

    covered_source = removed_ids | updated_ids
    covered_target = inserted_ids | updated_target_ids

    out: list[dict[str, t.Any]] = []
    for edit in edits:
        if isinstance(edit, Remove):
            node = edit.expression
            if not _has_covered_ancestor(node, covered_source):
                out.append(_remove_edit(source_paths[id(node)], node))
        elif isinstance(edit, Insert):
            node = edit.expression
            if not _has_covered_ancestor(node, covered_target):
                parent = matched[id(node.parent)]
                out.append(
                    {
                        "type": INSERT,
                        "parent": _json_path(source_paths[id(parent)]),
                        "parent_fingerprint": _fingerprint(parent),
                        "arg_key": node.arg_key,
                        "index": node.index,
                        "expression": serde.dump(node),
                    }
                )
        elif isinstance(edit, Update):
            node = edit.source
            if not _has_covered_ancestor(node, covered_source):
                out.append(_update_edit(source_paths[id(node)], node, edit.target))

    # Moves are processed shallowest first, so that a node whose ancestor's move already
    # carries it to its destination doesn't get a redundant (and potentially conflicting) move
    accepted_moves: dict[int, exp.Expr] = {}
    for edit in sorted(move_candidates.values(), key=lambda e: len(source_paths[id(e.source)])):
        node, target_node = edit.source, edit.target
        if _has_covered_ancestor(node, updated_ids):
            continue
        destination = target_node.parent
        if destination is None or _is_covered(destination, covered_target):
            # The destination subtree is inserted or replaced wholesale, so it already
            # carries this node's content; the source node can simply be dropped
            if not _has_covered_ancestor(node, covered_source):
                out.append(_remove_edit(source_paths[id(node)], node))
            continue
        if _is_redundant_move(node, target_node, matched, accepted_moves, covered_source):
            continue
        accepted_moves[id(node)] = target_node
        parent = matched[id(destination)]
        out.append(
            {
                "type": MOVE,
                "path": _json_path(source_paths[id(node)]),
                "fingerprint": _fingerprint(node),
                "parent": _json_path(source_paths[id(parent)]),
                "parent_fingerprint": _fingerprint(parent),
                "arg_key": target_node.arg_key,
                "index": target_node.index,
            }
        )

    result = Changeset(edits=out)

    # The diff's edit script is heuristic, so verify that the changeset actually replays;
    # if it doesn't, fall back to replacing the whole tree
    replayed = source.copy()
    try:
        result.apply(replayed)
        replays = replayed == target
    except Exception:
        replays = False
    if not replays:
        result = Changeset(edits=[_update_edit((), source, target)])

    return result


def _is_redundant_move(
    node: exp.Expr,
    target_node: exp.Expr,
    matched: dict[int, exp.Expr],
    accepted_moves: dict[int, exp.Expr],
    covered_source: set[int],
) -> bool:
    ancestor = node.parent
    while ancestor is not None:
        anchor = accepted_moves.get(id(ancestor))
        if anchor is not None:
            # The node rides along with its moved ancestor; the move is redundant if the
            # node already ends up at the target's position relative to that ancestor
            return _relpath(node, ancestor) == _relpath(target_node, anchor)
        ancestor = ancestor.parent
    # No ancestor moves, so the node stays put unless an ancestor is removed or updated
    return (
        not _has_covered_ancestor(node, covered_source)
        and matched.get(id(target_node.parent)) is node.parent
        and node.arg_key == target_node.arg_key
        and node.index == target_node.index
    )


def _relpath(node: exp.Expr, ancestor: exp.Expr) -> Path | None:
    path = []
    current: exp.Expr | None = node
    while current is not ancestor:
        if current is None:
            return None
        path.append((current.arg_key, current.index))
        current = current.parent
    return t.cast("Path", tuple(reversed(path)))


def apply(tree: exp.Expr, changeset: Changeset | dict[str, t.Any]) -> exp.Expr:
    """
    Replays a changeset on a tree, mutating it in place.

    Every node referenced by the changeset is resolved and checked against its content
    fingerprint before anything is mutated: if the tree no longer matches the changeset's
    source tree, a ChangesetConflictError is raised and the tree is left untouched.

    Args:
        tree: the expression to patch, typically a re-parse of the changeset's source tree.
        changeset: the Changeset (or its dict form) to apply.

    Returns:
        the mutated tree, or the new root node if the changeset replaces the whole tree
        with a node of a different type.
    """
    if isinstance(changeset, dict):
        changeset = Changeset.from_dict(changeset)

    nodes = _index_tree(tree)
    resolved: dict[tuple[int, str], exp.Expr] = {}
    conflicts: list[Conflict] = []

    for i, edit in enumerate(changeset.edits):
        for path_key, fingerprint_key in (
            ("path", "fingerprint"),
            ("parent", "parent_fingerprint"),
        ):
            if path_key in edit:
                path = _to_path(edit[path_key])
                node = nodes.get(path)
                actual = _fingerprint(node) if node is not None else None
                if node is None or actual != edit[fingerprint_key]:
                    conflicts.append(
                        Conflict(path=path, expected=edit[fingerprint_key], actual=actual)
                    )
                else:
                    resolved[(i, path_key)] = node

    if conflicts:
        raise ChangesetConflictError(conflicts)

    updates: list[tuple[int, dict]] = []
    removes: list[tuple[int, dict]] = []
    inserts: list[tuple[int, dict]] = []
    moves: list[tuple[int, dict]] = []
    buckets = {UPDATE: updates, REMOVE: removes, INSERT: inserts, MOVE: moves}
    for i, edit in enumerate(changeset.edits):
        buckets[edit["type"]].append((i, edit))

    for i, edit in updates:
        node = resolved[(i, "path")]
        new_node = t.cast(exp.Expr, serde.load(edit["expression"]))
        parent = node.parent
        if parent is None:
            if type(node) is not type(new_node):
                # A root replacement is always the only edit in a changeset
                return new_node
            node.args.clear()
            for arg_key, value in new_node.args.items():
                node.set(arg_key, value)
        else:
            parent.set(t.cast(str, node.arg_key), new_node, node.index)

    # Detach before attaching, so that final list positions can be used for insertions
    for i, edit in moves:
        resolved[(i, "path")].pop()
    for i, edit in removes:
        resolved[(i, "path")].pop()

    attachments = [
        (
            resolved[(i, "parent")],
            edit["arg_key"],
            edit["index"],
            t.cast(exp.Expr, serde.load(edit["expression"])),
        )
        for i, edit in inserts
    ]
    attachments += [
        (resolved[(i, "parent")], edit["arg_key"], edit["index"], resolved[(i, "path")])
        for i, edit in moves
    ]

    by_parent: dict[int, list[tuple[exp.Expr, str, int | None, exp.Expr]]] = defaultdict(list)
    for attachment in attachments:
        by_parent[id(attachment[0])].append(attachment)

    for group in by_parent.values():
        for parent, arg_key, index, node in sorted(group, key=lambda a: (a[2] is None, a[2])):
            values = parent.args.get(arg_key)
            if index is None:
                parent.set(arg_key, node)
            elif isinstance(values, list) and index < len(values):
                parent.set(arg_key, node, index, overwrite=False)
            else:
                parent.append(arg_key, node)

    return tree


def _fingerprint(node: exp.Expr) -> str:
    return hashlib.sha256(node.sql(comments=False).encode("utf-8")).hexdigest()


def _index_tree(root: exp.Expr) -> dict[Path, exp.Expr]:
    index: dict[Path, exp.Expr] = {(): root}
    stack: list[tuple[exp.Expr, Path]] = [(root, ())]
    while stack:
        node, path = stack.pop()
        for arg_key, value in node.args.items():
            if isinstance(value, exp.Expr):
                children: list[tuple[exp.Expr, int | None]] = [(value, None)]
            elif type(value) is list:
                children = [(item, i) for i, item in enumerate(value)]
            else:
                continue
            for item, i in children:
                if isinstance(item, exp.Expr):
                    child_path = path + ((arg_key, i),)
                    index[child_path] = item
                    stack.append((item, child_path))
    return index


def _is_covered(node: exp.Expr, covered: set[int]) -> bool:
    return id(node) in covered or _has_covered_ancestor(node, covered)


def _has_covered_ancestor(node: exp.Expr, covered: set[int]) -> bool:
    current = node.parent
    while current is not None:
        if id(current) in covered:
            return True
        current = current.parent
    return False


def _update_edit(path: Path, source: exp.Expr, target: exp.Expr) -> dict[str, t.Any]:
    return {
        "type": UPDATE,
        "path": _json_path(path),
        "fingerprint": _fingerprint(source),
        "expression": serde.dump(target),
    }


def _remove_edit(path: Path, node: exp.Expr) -> dict[str, t.Any]:
    return {"type": REMOVE, "path": _json_path(path), "fingerprint": _fingerprint(node)}


def _json_path(path: Path) -> list[list[t.Any]]:
    return [[arg_key, index] for arg_key, index in path]


def _to_path(steps: list[list[t.Any]]) -> Path:
    return tuple((arg_key, index) for arg_key, index in steps)


def _format_path(path: Path) -> str:
    if not path:
        return "<root>"
    return ".".join(
        f"{arg_key}[{index}]" if index is not None else arg_key for arg_key, index in path
    )
