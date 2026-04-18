import tensorflow as tf 
# it is the engine which run trained model(deseasemodel.h5)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2
# mobilenetv2 is a pretrained model trained on imagenet dataset

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# preprocess_input prepares images the way mobilenet expects them

from tensorflow.keras import layers, models

IMG_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 10


datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=20,

    zoom_range=0.2,

    horizontal_flip=True,
    

    validation_split=0.2
    
)

# reads image from folder for training
train_data = datagen.flow_from_directory(

    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

# it validate data from train
val_data = datagen.flow_from_directory(

# check the size, batch ,mode 
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"

)

# transfer learning (MobileNetV2)
base_model = MobileNetV2(

    weights="imagenet",
    # loads weights that already trained

    include_top=False,
    # removes top layer so we can add our own of leaf

    input_shape=(IMG_SIZE,IMG_SIZE,3)
)


base_model.trainable = False
# freezing base model so its weights dont change during training


# custom top of base model
x = base_model.output

x = layers.GlobalAveragePooling2D()(x)
# convert 2d image to vector

x = layers.Dense(128,activation="relu")(x)

x = layers.Dropout(0.3)(x)

output = layers.Dense(train_data.num_classes,activation="softmax")(x)
# softmax = raw data to percentage 100%

model = models.Model(

    inputs=base_model.input,
    outputs=output
)

model.compile(

    optimizer="adam",
    #adjusts learning rate automatically

    loss="categorical_crossentropy",
    #used for multi class classification

    metrics=["accuracy"]
)

# train model
history = model.fit(

    train_data,
    validation_data=val_data,
    epochs=EPOCHS
    # trains 10 rounds on full dataset
)

model.save("model/diseasemodel.h5")
# saves the model

print("Model training completed")