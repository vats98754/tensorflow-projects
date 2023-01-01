from tensorflow import keras

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data=keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

vocab_size = 10000
model = keras.models.Sequential([
	keras.layers.Embedding(vocab_size, 16),
	keras.layers.GlobalAveragePooling1D(),
	keras.layers.Dense(16, activation='relu'),
	keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(
	loss='binary_crossentropy',
	optimizer=keras.optimizers.Adam(),
	metrics=['accuracy']
)

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(
	partial_x_train,
	partial_y_train,
	epochs=40,
	batch_size=512,
	validation_data=(x_val, y_val),
	verbose=1
)

results = model.evaluate(test_data, test_labels)
print(results)
