from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = '../ML/rps/training/'
training_datagen = ImageDataGenerator(1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical'
)

TESTING_DIR = '../ML/rps/testing/'
testing_datagen = ImageDataGenerator(1./255)

testing_generator = testing_datagen.flow_from_directory(
	TESTING_DIR,
	target_size=(150,150),
	class_mode='categorical'
)

model = keras.models.Sequential([
	keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
	keras.layers.MaxPooling2D(2,2),
	keras.layers.Conv2D(64, (3,3), activation='relu'),
	keras.layers.MaxPooling2D(2,2),
	keras.layers.Conv2D(128, (3,3), activation='relu'),
	keras.layers.MaxPooling2D(2,2),
	keras.layers.Conv2D(128, (3,3), activation='relu'),
	keras.layers.MaxPooling2D(2,2),
	# Feed into Dense Neural Network (DNN)
	keras.layers.Flatten(),
	keras.layers.Dropout(0.5),  # To somewhat reduce overfitting but mainly to increase efficiency
	keras.layers.Dense(512, activation='relu'),
	keras.layers.Dense(3, activation='softmax')  # 3 neurons because r,p,s are 3 classes (softmax used for distinction)
])

model.compile(
	loss="categorical_crossentropy",
	optimizer="rmsprop",
	metrics=['accuracy']
)
model.summary()

history = model.fit(
	train_generator,
	epochs=25,
	validation_data=testing_generator,
	verbose=1
)

classes = model.predict('../ML/rps/validation/', batch_size=10)
