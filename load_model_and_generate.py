import tensorflow as tf
import numpy as np
import os
import shutil
path_to_file = tf.keras.utils.get_file('all.txt', 'https://raw.githubusercontent.com/ayyucedemirbas/Star-Trek-TOS-Transcripts/main/all.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))

# Create a mapping from characters to numbers and vice versa
char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)
model = tf.keras.models.load_model('Model_3_LSTM.h5')
model.summary()
def generate_text(model, start_string):
  # Number of characters to generate
  num_generate = 5000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []
  temperature = 0.7

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      predictions = tf.squeeze(predictions, 0)

      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))
print(generate_text(model, start_string=u"Kirk"))