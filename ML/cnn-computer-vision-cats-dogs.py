import shutil

from tensorflow import keras
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import random

try:
	os.mkdir('../ML/cats-v-dogs')
	os.mkdir('../ML/cats-v-dogs/training')
	os.mkdir('../ML/cats-v-dogs/testing')
	os.mkdir('../ML/cats-v-dogs/training/cats')
	os.mkdir('../ML/cats-v-dogs/training/dogs')
	os.mkdir('../ML/cats-v-dogs/testing/cats')
	os.mkdir('../ML/cats-v-dogs/testing/dogs')
except OSError:
	pass


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
	files = []
	for filename in os.listdir(SOURCE):
		file = SOURCE + filename
		if os.path.getsize(file) > 0:
			files.append(filename)
		else:
			print(filename + " has 0 length, so ignored.")

	training_length = int(len(files) * SPLIT_SIZE)
	shuffled_set = random.sample(files, len(files))
	training_set = shuffled_set[0:training_length]
	testing_set = shuffled_set[training_length:]

	for filename in training_set:
		this_file = SOURCE + filename
		destination = TRAINING + filename
		shutil.copyfile(this_file, destination)

	for filename in testing_set:
		this_file = SOURCE + filename
		destination = TESTING + filename
		shutil.copyfile(this_file, destination)


CAT_SOURCE_DIR = 'horses-v-humans/PetImages/Cat/'
TRAINING_CATS_DIR = 'horses-v-humans/cats-v-dogs/training/cats/'
TESTING_CATS_DIR = 'horses-v-humans/cats-v-dogs/testing/cats/'
DOG_SOURCE_DIR = 'horses-v-humans/PetImages/Dog/'
TRAINING_DOGS_DIR = 'horses-v-humans/cats-v-dogs/training/dogs/'
TESTING_DOGS_DIR = 'horses-v-humans/cats-v-dogs/testing/dogs/'

split_size = 0.9  # 90% of the data randomly assigned for training, other 10% randomly assigned for testing
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

model = keras.models.Sequential(
	keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # 3 for color channels
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Conv2D(32, (3, 3), activation='relu'),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Conv2D(64, (3, 3), activation='relu'),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Flatten(),
	keras.layers.Dense(512, activation='relu'),
	keras.layers.Dense(1, activation='sigmoid')  # because binary classification
)
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])
model.summary()

TRAINING_DIR = '/tmp/cats-v-dogs/training/'
VALIDATION_DIR = '/tmp/cats-v-dogs/testing/'

train_datagen = ImageDataGenerator(1./255)
train_generator = train_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150, 150),
	batch_size=250,
	class_mode='binary'
)

test_datagen = ImageDataGenerator(1./255)
validation_generator = test_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150, 150),
	batch_size=250,
	class_mode='binary'
)

history = model.fit(
	train_generator,
	steps_per_epoch=90,  # dataset_size = 22500, we chose batch_size = 250, so 22500 / 250 = 90
	epochs=15,
	validation_data=validation_generator,
	validation_steps=10,  # dataset_size =  2500, batch_size = 250, so 2500 / 250 = 10
	verbose=2
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get the number of epochs to display progress over each epoch

plt.plot(epochs, acc, 'r', 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', 'Training Loss')
plt.plot(epochs, val_loss, 'b', 'Validation Loss')
plt.title('Training and Validation Loss')
plt.figure()
#
# # Test model with Google Colab
# import numpy as np
# from google.colab import files
# from keras.preprocessing import image
#
# uploaded = files.upload()
#
# for fn in uploaded.keys():
# 	# predicting images
# 	path = '/content/' + fn
# 	img = image.load_img(path, target_size=(150,150))
# 	x = image.img_to_array(img)
# 	x = np.expand_dims(x, axis=0)
#
# 	images = np.vstack([x])
# 	classes = model.predict(images, batch_size=10)
# 	print(classes[0])
# 	if classes[0] > 0.5:
# 		print(fn + " is a dog!")
# 	else:
# 		print(fn + " is a cat!")
