import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

EMBED_DIM = 128
HEADS = 4
FF_UNITS = 128
VOCAB_SIZE = 100
SEQUENCE_LEN = 50
CLASS_COUNT = 3
BLOCK_COUNT = 5

def load_data(filepath):
    data = pd.read_csv(filepath)
    data.dropna(inplace=True)
    return data['password'].values, data['strength'].values

def preprocess_data(passwords, strengths):
    passwords = [str(password) for password in passwords]
    label_encoder = LabelEncoder()
    encoded_strengths = label_encoder.fit_transform(strengths)
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, char_level=True, oov_token='UNK')
    tokenizer.fit_on_texts(passwords)
    sequences = tokenizer.texts_to_sequences(passwords)
    padded_sequences = pad_sequences(sequences, maxlen=SEQUENCE_LEN)
    return padded_sequences, encoded_strengths, tokenizer

def split_data(inputs, labels):
    return train_test_split(inputs, labels, test_size=0.2, random_state=42)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def get_positional_encodings():
    angle_rads = np.arange(SEQUENCE_LEN)[:, np.newaxis] / np.power(10000, (2 * (np.arange(EMBED_DIM)[np.newaxis, :]//2)) / EMBED_DIM)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

def transformer_encoder(input_tensor):
    attention_out = layers.MultiHeadAttention(num_heads=HEADS, key_dim=EMBED_DIM)(input_tensor, input_tensor)
    out1 = layers.LayerNormalization(epsilon=1e-6)(input_tensor + attention_out)
    feed_forward_out = layers.Dense(FF_UNITS, activation="relu")(out1)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + layers.Dense(EMBED_DIM)(feed_forward_out))
    return out2

def construct_transformer():
    inputs = layers.Input(shape=(SEQUENCE_LEN,))
    embeddings = layers.Embedding(VOCAB_SIZE, EMBED_DIM)(inputs) + get_positional_encodings()
    x = embeddings
    for _ in range(BLOCK_COUNT):
        x = transformer_encoder(x)
    x = layers.Bidirectional(layers.LSTM(EMBED_DIM, return_sequences=True))(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(CLASS_COUNT, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

class SaveModelAtEpoch(keras.callbacks.Callback):
    def __init__(self, epoch_to_save, save_path):
        self.epoch_to_save = epoch_to_save
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.epoch_to_save:  # epochs are zero-indexed
            self.model.save(self.save_path)
            print(f"Model saved at epoch {self.epoch_to_save} to {self.save_path}")

def train_transformer(model, train_inputs, train_labels, val_inputs, val_labels):
    checkpoint = keras.callbacks.ModelCheckpoint("best_transformer_model.h5", save_best_only=True, monitor="val_accuracy")
    save_at_epoch_4 = SaveModelAtEpoch(4, "model_epoch_04.h5")
    early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    try:
        model.fit(
            train_inputs, train_labels, validation_data=(val_inputs, val_labels), 
            epochs=20, batch_size=64,
            callbacks=[checkpoint, save_at_epoch_4, early_stop]
        )
    except KeyboardInterrupt:
        print("Training interrupted by user. Latest model saved at the last checkpoint.")
        model.save('transformer_1.h5')

def main():
    passwords, strengths = load_data('data.csv')
    inputs, labels, tokenizer = preprocess_data(passwords, strengths)
    train_inputs, val_inputs, train_labels, val_labels = split_data(inputs, labels)
    transformer = construct_transformer()
    train_transformer(transformer, train_inputs, train_labels, val_inputs, val_labels)
    transformer.save("final_transformer_model.h5")

if __name__ == "__main__":
    main()

