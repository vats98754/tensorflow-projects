import keras.utils
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

tokenizer = Tokenizer()

data = open('../NLP/irish_poem.txt').read()
corpus = data.lower().split('\n')

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # To account for some Out-Of-Vocabulary token

input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
ys = keras.utils.to_categorical(labels, num_classes=total_words)

# The following model is weak, try to add more functional layers as necessary
model = tf.keras.Sequential([
	tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1),
	# input_dims is proportional to the variety of words
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
	tf.keras.layers.Dense(total_words, activation='softmax')
])
adam = tf.optimizers.Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

history = model.fit(xs, ys, epochs=100, verbose=1)

seed_text = 'I made a poetry machine'
next_words = 20

for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=0)
	output_word = ''
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word

print(seed_text)
