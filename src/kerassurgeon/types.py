from typing import Any, Protocol, TypeAlias

import tensorflow as tf

Masks: TypeAlias = Any
Node: TypeAlias = Any
Inputs: TypeAlias = tf.Tensor | list[tf.Tensor]


class ModificationFunction(Protocol):
    # pylint: disable=R0903
    def __call__(
        self,
        node: Node,
        inputs: Inputs,
        input_masks: Masks,
        **kwargs,
    ) -> None:
        ...
