import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

sentences = [
	'I love my dog',
	'I love my cat'
]

tokenizer = Tokenizer(num_words=100)  # id's of the 100 most frequent words used
tokenizer.fit_on_texts(sentences)  # tokenizes each word in each sentence; greater frequency => lower token
word_index = tokenizer.word_index
print(word_index)
