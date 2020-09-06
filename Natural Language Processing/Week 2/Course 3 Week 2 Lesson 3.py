import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised = True)

#Use the presetted tokenizer
train_data, test_data = imdb['train'], imdb['test']
tokenizer = info.features['text'].encoder
print(tokenizer.subwords)

#Select one data string and see what they are
sample_string = 'Tensorflow, from basics to mastery'
tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))
original_string = tokenizer.encode(tokenized_string)
print('The original string is {}'.format(original_string))

for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer.decode([ts])))

#Set Training and Testing dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_data.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))

#Build Model
embedding_dim = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation= 'relu'),
    tf.keras.layers.Dense(1, activation= 'sigmoid')])
model.summery()

#Compile and train model
num_epochs = 10
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(train_data, epochs=num_epochs, validation_data = test_data)

#Show training history
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

#Export Results data to http://projector.tensorflow.org/
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, tokenizer.vocab_size):
  word = tokenizer.decode([word_num])
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

'''
#Download from Google colab
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')
'''
