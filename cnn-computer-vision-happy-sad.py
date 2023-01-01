import tensorflow as tf
from tensorflow import keras
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator


class MyCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		if logs.get('accuracy') > 0.998:
			print('\nReached 99.8% accuracy so cancelling training!')
			self.model.stop_training = True


callbacks = MyCallback()

model = keras.models.Sequential([
	keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Conv2D(32, (3, 3), activation='relu'),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Conv2D(32, (3, 3), activation='relu'),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Flatten(),
	keras.layers.Dense(512, activation='relu'),
	keras.layers.Dense(1, activation='sigmoid')  # Since this is a binary classification, one output node will suffice
])
model.compile(
	loss='binary_crossentropy',
	optimizer=RMSprop(),
	metrics=['acc']
)

train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
	"../ML/happy-v-sad/",
	target_size=(150,150),
	batch_size=10,
	class_mode='binary'
)

model.summary()
history = model.fit(
	train_generator,
	steps_per_epoch=8,
	epochs=15,
	verbose=1,
	# callbacks=[callbacks] # Only for Colab
)