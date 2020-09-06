import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
	'I love my dog',
	'I love my cat',
	'You love my dog!',
	'Do you think my dog is amazing?'
]

#Set the words in sentences into a dictionary of word with index
tokenizer = Tokenizer(num_words = 100, oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding = 'post', maxlen=5)

#Print out index and padded sentences
print("\nWord Index = ", word_index)
print(sentences[2])
print("\nSequences = ", sequences)
print("\nPadded Sequences:")
print(padded)

#Test Data
test_data = [
	'I really love my dog',
	'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

padded = pad_sequences(test_seq, maxlen = 10)
print("\nPadded Test Sequence: ")
print(padded)

'''
#Download 
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \
    -O /tmp/sarcasm.json
  
import json

with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)


sentences = [] 
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

'''