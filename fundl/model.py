from copy import deepcopy
from typing import List
import tensorflow as tf

from .config import FunDLBlockConfig, FunDLObjective


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
        hidden_layers: List[FunDLBlockConfig],
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
        self, inputs, instruction: FunDLBlockConfig
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
        model = tf.keras.Model(inputs=inputs, outputs=out, name="FunDL")
        return model

    def __build(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(self.__input_shape, 3), name="input")
        layer_config = self.__hidden_layers[0]
        if not layer_config.dropout_rate:
            layer_config.dropout_rate = (len(self.__hidden_layers)) * 0.1
        x = self.__add_block(inputs, layer_config)
        for idx, layer_config in enumerate(self.__hidden_layers[1:]):
            if not layer_config.dropout_rate:
                layer_config.dropout_rate = (len(self.__hidden_layers) - idx) * 0.1
                x = self.__add_block(x, layer_config)
        model = self.__finalise(inputs, x)
        return model

    def call(self, inputs, training=False):
        """
        Call the model
        """
        # inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        return self.__model(inputs)

    def summary(self, **kwargs):
        self.__model.summary(**kwargs)
