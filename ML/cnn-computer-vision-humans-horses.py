from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop

train_dir = "horses-v-humans/training/"
validation_dir = "horses-v-humans/testing/"

train_datagen = ImageDataGenerator(rescale=1./255)  # Normalizes the training image data
train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(300, 300),
	batch_size=128,
	class_mode='binary'  # Since we only have 2 classes (horses and humans), use 'binary'. Else, use 'categorical'.
)

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
	validation_dir,
	target_size=(300, 300),
	batch_size=32,
	class_mode='binary'  # Since we only have 2 classes (horses and humans), use 'binary'. Else, use 'categorical'.
)

model = keras.models.Sequential([
	keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Conv2D(32, (3, 3), activation='relu'),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Conv2D(64, (3, 3), activation='relu'),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Flatten(),
	keras.layers.Dense(512, activation='relu'),
	keras.layers.Dense(1, activation='sigmoid')  # Since this is a binary classification, one output node will suffice
])

model.compile(
	loss='binary_crossentropy',
	optimizer=RMSprop(lr=0.001),  # Learning rate is 0.001 for gradient descent
	metrics=['accuracy']
)
model.summary()  # Prints the details within each layer of the CNN
history = model.fit(
	train_generator,
	epochs=15,
	validation_data=validation_generator,
	steps_per_epoch=8,  # usually, steps = dataset_size / batch_size ~ 1024/128
	validation_steps=8,
	verbose=2
)

#
# # Test model with Google Colab (ONLY WORKS WITH JUPYTER NOTEBOOKS)
# import numpy as np
# from google.colab import files
# from keras.preprocessing import image
#
# uploaded = files.upload()
#
# for fn in uploaded.keys():
# 	# predicting images
# 	path = '/content/' + fn
# 	img = image.load_img(path, target_size=(300,300))
# 	x = image.img_to_array(img)
# 	x = np.expand_dims(x, axis=0)
#
# 	images = np.vstack([x])
# 	classes = model.predict(images, batch_size=10)
# 	print(classes[0])
# 	if classes[0] > 0.5:
# 		print(fn + " is a human!")
# 	else:
# 		print(fn + " is a horse!")
