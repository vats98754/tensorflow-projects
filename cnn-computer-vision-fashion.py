import tensorflow as tf
from tensorflow import keras


class MyCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		if logs.get('accuracy') > 0.998:
			print('\nReached 99.8% accuracy so cancelling training!')
			self.model.stop_training = True


callbacks = MyCallback()
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

# TODO: This method of Flattening immediately only works on images with the same format (centering, grayscale,
#  28pxx28px, etc) as the training data. For greater versatility, use CNNs that instead identify features in images.
model = keras.Sequential([
	keras.layers.Conv2D(64, (3, 3), activation='relu',
	                    input_shape=(28, 28, 1)),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Conv2D(64, (3, 3), activation='relu'),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Flatten(),
	keras.layers.Dense(128, activation='relu'),  # 128 = 2*2*64 (image area * number of filters in CNN)
	keras.layers.Dense(10, activation='softmax')
])

model.compile(
	optimizer=tf.compat.v1.train.AdamOptimizer(),
	loss='sparse_categorical_crossentropy',
	metrics='accuracy'
)
model.summary()  # Prints the details within each layer of the CNN
model.fit(train_images, train_labels, epochs=100, callbacks=[callbacks])
test_loss, test_acc = model.evaluate(test_images, test_labels)

# FOR TESTING: predictions = model.predict(my_images)
