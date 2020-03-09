"""
TensorFlow example of predicting whether a movie review is positive or negative
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

""" NLP with HUB

# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
# print(train_examples_batch)
# print(train_labels_batch)

# embedding for transfer learning
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

# model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

# print(model.summary())

# 'compile' model - add optimizer and loss functions
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# test
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
"""

from tensorflow import keras
tfds.disable_progress_bar()

# load in dataset
(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k', 
    # Return the train/test datasets as a tuple.
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised=True,
    # Also return the `info` structure. 
    with_info=True)

# init encoder to go from string to int
encoder = info.features['text'].encoder
print (f'Vocabulary size: {encoder.vocab_size}')
sample_string = 'TensorFlow is awesome!'
encoded_string = encoder.encode(sample_string)
decoded_string = encoder.decode(encoded_string)
print(f'Encoded: {encoded_string}\nDecoded: {decoded_string}\n'
      f'Matching?: {sample_string==decoded_string}')

# see tokens
for token in encoded_string:
    print(f'{token} ---> {encoder.decode([token])}')

# looking at the data
for train_example, train_label in train_data.take(1):
    print(f'Encoded text: {train_example[:10].numpy()}')
    print(f'Label: {train_label.numpy()}')
print(encoder.decode(train_example[:10]))

# data prep
BUFFER_SIZE = 1000

train_batches = train_data.shuffle(BUFFER_SIZE).padded_batch(32)
test_batches = test_data.padded_batch(32)

for example_batch, label_batch in train_batches.take(2):
    print(f'Batch shape: {example_batch.shape}')
    print(f'Label shape: {label_batch.shape}')

# model
model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(1)
])
print(model.summary())

# compile model
model.compile(
    optimizer='adam',
    loss=tf.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# train
history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches,
    validation_steps=30
)

# test
loss, accuracy = model.evaluate(test_batches)
print(f'Loss: {loss}\nAccuracy: {accuracy}')

# model analyzation
import matplotlib.pyplot as plt

history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
