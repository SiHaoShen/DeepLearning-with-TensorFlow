import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from urllib.request import urlopen
import numpy as np

#Download sarcasm data
filedata = urlopen('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt')
datatowrite = filedata.read()

#Load Data and Separate Training/Test Data
with open("C:/Github/TensorFlow/Course_Material/tmp/irish-lyrics-eof.txt", "wb") as f:
    f.write(datatowrite)

#Set the example data
tokenizer = Tokenizer()
data = open('C:/Github/TensorFlow/Course_Material/tmp/irish-lyrics-eof.txt').read()
corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

#Build a dictionary of corpus text
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences]) #Maximal of the sentence length
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding= 'pre'))

xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

#Build a model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1)) #Tune 64 Dim
model.add(Bidirectional(LSTM(150))) #Tune LSTM units
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose =1)

#Show the history of accuracy of the trained model
import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

plot_graphs(history, 'accuracy')

#Test out the model
seed_text = "I've got a bad feeling about this"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding = 'pre')
    predicted = model.predict_classes(token_list, verbose = 0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)