import json
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

data_store = []

for line in open('sarcasm.json', 'r'):
	data_store.append(json.loads(line))

sentences = []
labels = []
urls = []
for item in data_store:
	sentences.append(item['headline'])
	labels.append(item['is_sarcastic'])
	urls.append(item['article_link'])

# TODO: Using a single dataset, we can use the first n data for training and the remaining data for testing
training_size = 20000  # Set the training size as necessary
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# The following hyperparameters need to be global, for each of training, testing, and custom data
vocab_size = 10000
max_length = 100
padding_type = 'post'
trunc_type = 'post'
embedding_dim = 16  # how many nuances/directions can a word have

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")  # choose the appropriate top n noteworthy tokens
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

print(testing_padded[0])
print(testing_padded.shape)  # prints (row_num, col_num) of the padded matrix

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
	tf.keras.layers.GlobalAveragePooling1D(),
	tf.keras.layers.Dense(24, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)

# For custom prediction, we can use produce our own test data
custom_sentences = [
	'granny starting to fear spiders in the garden might be real',
	'the weather today is bright and sunny'
]

custom_sequences = tokenizer.texts_to_sequences(custom_sentences)

custom_padded = pad_sequences(custom_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(custom_padded))