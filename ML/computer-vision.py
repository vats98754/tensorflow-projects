import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# TODO: This method of Flattening immediately only works on images with the same format (centering, grayscale,
#  28pxx28px, etc) as the training data. For greater versatility, use CNNs that instead identify features in images.
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
	optimizer=tf.compat.v1.train.AdamOptimizer(),
	loss='sparse_categorical_crossentropy',
	metrics='accuracy'
)

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

# FOR TESTING: predictions = model.predict(my_images)
