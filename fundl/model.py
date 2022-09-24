from copy import deepcopy
from typing import List
import tensorflow as tf

from .config import FunDLnBlockConfig, FunDLObjective


class DenseDropoutBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int = 32,
        activation: str = "relu",
        add_dropout: bool = True,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super(DenseDropoutBlock, self).__init__(**kwargs)
        name = kwargs.get("name")
        self.__add_dropout = add_dropout
        self.dense = tf.keras.layers.Dense(
            units=units, activation=activation, name=f"{name}_dense"
        )
        self.dropout = tf.keras.layers.Dropout(
            rate=dropout_rate, name=f"{name}_dropout"
        )

    def call(self, inputs: tf.keras.Input, training: bool = False):
        x = self.dense(inputs)
        if training and self.__add_dropout:
            x = self.dropout(x)
        return x


class FunDL(tf.keras.Model):
    def __init__(
        self,
        objective: FunDLObjective,
        input_shape: int,
        hidden_layers: List[FunDLnBlockConfig],
        output_size: int,
    ):
        super(FunDL, self).__init__()
        self.__objective = FunDLObjective(objective)
        self.__input_shape = input_shape
        self.__hidden_layers = hidden_layers
        self.__output_size = output_size
        self.__validate()
        self.__model = self.__build()

    def __validate(self):
        if self.__input_shape < 1 and not isinstance(self.__input_shape, int):
            raise ValueError(f"Invalid 'input_shape': {self.__input_shape}")
        if len(self.__hidden_layers) < 1:
            raise ValueError(
                f"Number of hidden layers needs to be greater or equal 1, not {len(self.__hidden_layers)}"
            )
        if self.__output_size < 1 and not isinstance(self.__output_size, int):
            raise ValueError(f"Invalid 'output_size': {self.__output_size}")

    def __add_block(
        self, inputs, instruction: FunDLnBlockConfig
    ) -> tf.keras.layers.Layer:
        block = DenseDropoutBlock(
            units=instruction.units,
            activation=instruction.activation,
            add_dropout=instruction.add_dropout,
            dropout_rate=instruction.dropout_rate,
            name=instruction.name,
        )
        return block(inputs)

    def __finalise(self, inputs, x) -> tf.keras.Model:
        activation = None
        if self.__objective is FunDLObjective.CLASSIFICATION:
            activation = "softmax"
        out = tf.keras.layers.Dense(
            units=self.__output_size, activation=activation, name="output"
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=out, name=self.__class__.name)
        return model

    def __build(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.__input_shape,), name="input")
        x = deepcopy(inputs)
        for idx, layer_config in enumerate(self.__hidden_layers):
            if not layer_config.dropout_rate:
                layer_config.dropout_rate = (len(self.__hidden_layers) - idx) * 0.1
            x = self.__add_block(x, layer_config)
        model = self.__finalise(inputs, x)
        return model

    def call(self, inputs, training=False):
        """
        Call the model
        """
        __model = self.__build()
        return __model(inputs)

    def summary(self, **kwargs):
        self.__model.summary(**kwargs)


if __name__ == "__main__":
    hidden_layers = [
        FunDLnBlockConfig(
            name="hidden01",
            units=16,
            activation="relu",
            dropout_rate=None,
        ),
        FunDLnBlockConfig(
            name="hidden02",
            units=32,
            activation="relu",
            dropout_rate=None,
        ),
        FunDLnBlockConfig(
            name="hidden03",
            units=16,
            activation="relu",
            dropout_rate=None,
        ),
        FunDLnBlockConfig(
            name="hidden04",
            units=8,
            activation="relu",
            add_dropout=False,
            dropout_rate=None,
        ),
    ]
    input_shape = 12
    output_size = 2
    model = FunDL(
        objective=FunDLObjective("classification"),
        input_shape=input_shape,
        hidden_layers=hidden_layers,
        output_size=output_size,
    )

    print(model.summary(line_length=80, show_trainable=True))
