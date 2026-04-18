# basic cnn train_model.py:
import tensorflow as tf 
# it is the engine which runs and builds the model


from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers, models

IMG_SIZE = 224

BATCH_SIZE = 16

datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2
)

train_data = datagen.flow_from_directory(

    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(

    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"

)

model = models.Sequential([

layers.Conv2D(32,(3,3),activation="relu",input_shape=(IMG_SIZE,IMG_SIZE,3)),
# first convolution layer extracts basic features
layers.MaxPooling2D(2,2),

layers.Conv2D(64,(3,3),activation="relu"),
# second layer extracts deeper features
layers.MaxPooling2D(2,2),

layers.Conv2D(128,(3,3),activation="relu"),
# third layer extracts complex patterns
layers.MaxPooling2D(2,2),

layers.Flatten(),
# converts 2d feature maps into 1d vector

layers.Dense(128,activation="relu"),
# fully connected layer
layers.Dense(train_data.num_classes,activation="softmax")
# output layer gives class probabilities

])

model.compile(

optimizer="adam",
# adam adjusts learning rate automatically
loss="categorical_crossentropy",
# used for multi class classification
metrics=["accuracy"]

)

model.fit(

    train_data,
    validation_data=val_data,
    epochs=5
    # train for 5 rounds on the dataset
)

model.save("model/diseasemodel.h5")
# saves the trained model to disk

print("Model training completed")