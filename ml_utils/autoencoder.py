import keras
from typing import Tuple


class Autoencoder(keras.Model):
    def __init__(self, hidden_struct: Tuple[int, str], output_layer: Tuple[int, str]):
        super(Autoencoder, self).__init__()
        self.encoder = keras.Sequential()
        self.encoder.add(keras.layers.Flatten())

        for layer in hidden_struct:
            latent_dim = layer[0]
            activation = layer[1]
            self.encoder.add(keras.layers.Dense(latent_dim, activation=activation))

        self.decoder = keras.Sequential()
        for layer in reversed(hidden_struct[:-1]):
            latent_dim = layer[0]
            activation = layer[1]
            self.decoder.add(keras.layers.Dense(latent_dim, activation=activation))
        self.decoder.add(keras.layers.Dense(output_layer[0], activation=output_layer[1]))

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_dim_ranges(input_dim: int, latent_ditm: int, num_layers: int):

    dim_ranges = []
    increment = int((input_dim - latent_ditm) / num_layers)

    minimum = latent_ditm + 1
    for layer in range(num_layers):
        maximum = minimum + increment
        if maximum > input_dim - 1:
            maximum = input_dim - 1
        dim_ranges.append([minimum, maximum])
        minimum += increment + 1

    return dim_ranges
