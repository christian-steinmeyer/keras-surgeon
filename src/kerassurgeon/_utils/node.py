import collections.abc
from typing import TypeVar, cast

from kerassurgeon._utils import layer as layer_utils

T = TypeVar("T")


def make_list_if_not(x: T | list[T]) -> list[T]:
    if not isinstance(x, list):
        if isinstance(x, collections.abc.Sequence) and not isinstance(x, str):
            return list(x)
        x = cast(T, x)
        return [x]
    return x


def node_indices(node) -> list[int]:
    return make_list_if_not(node.node_indices)


def inbound_layers(node):
    return make_list_if_not(node.inbound_layers)


def parent_nodes(node):
    try:
        return node.parent_nodes
    except AttributeError:
        return [
            layer_utils.inbound_nodes(inbound_layers(node)[i])[node_index]
            for i, node_index in enumerate(node_indices(node))
        ]
