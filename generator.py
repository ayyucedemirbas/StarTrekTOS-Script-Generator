import tensorflow as tf
import numpy as np
import os
import shutil
path_to_file = tf.keras.utils.get_file('all.txt', 'https://raw.githubusercontent.com/ayyucedemirbas/Star-Trek-TOS-Transcripts/main/all.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print('Length of Character {}'.format(len(text)))
print(text[:250])
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))
print(vocab)
# Create a mapping from characters to numbers and vice versa
char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
print("{} is mapped to {}".format(text[:10], text_as_int[:10]))
# Maximum sentence we are inputing to the RNN
seq_length =100
examples_per_epoch = len(text)//(seq_length+1)
print(examples_per_epoch)
# Creating dataset
# from_tensor_slices is like creating a generator for our dataset and is suitable for handling
# large datasets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
# .take is like iloc in pandas
for i in char_dataset.take(5):
    print(idx2char[i.numpy()])
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

def split_input_target(chunk):
    """
    This function generate input and target text from the given text.
    Input text does not contain last part and target doesnot contain first character
    """
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset
#Constants for model
vocab_size = len(vocab)
embedding_dim = 512
# Using half of rnn_units for LSTM
# Speed of training was reduced to half, so i can try 1024 units
rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences = True, stateful= True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.LSTM(rnn_units, return_sequences = True, stateful= True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model
model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

#Here we are chosing the next character randomly based on its probablity
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
#idk what this line does
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
sampled_indices
#Decoding what this means
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 5
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
