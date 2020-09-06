import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from urllib.request import urlopen
import matplotlib.pyplot as plt

#Setup Variables
vocab_size = 10000
embedding_dim = 16
max_length = 100 #120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000

#Download sarcasm data
filedata = urlopen('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json ')
datatowrite = filedata.read()

'''
#Download using google colab
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \
    -O /tmp/sarcasm.json
'''

#Load Data and Separate Training/Test Data
with open("Datapath", "r") as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

#Tokenize Training and Testing Data
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#Make Training and Testing data into array
training_padded = np.array(training_padded)
testing_padded = np.array(testing_padded)
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

#Build a Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)), #For LSTM layer, without GlobalAveragePooling
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.kayers.Dense(1, activation='sigmoid')])

'''
#Build a CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)), #For LSTM layer, without GlobalAveragePooling
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.kayers.Dense(1, activation='sigmoid')])
'''
#Compile the Model
model.compile(loss='binary_crossentropy', optimzer='adam', metrics=['accuracy'])
model.summary()


#Train the model
num_epochs = 30 #50
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose =2)
'''
#Show accuracy and loss on the training and validation dataset
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
'''

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_sentence(training_padded[0]))
print(training_sentences[2])
print(labels[2])

#Get the weight of the first layer
e= model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

#Write and Save the Output file
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

model.save("test.h5")