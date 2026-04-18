import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models


IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5


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

#BASIC CNN 
print("\n🔵 Training Basic CNN...")

basic_model = models.Sequential([
    layers.Conv2D(32,(3,3),activation="relu",input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128,activation="relu"),
    layers.Dense(train_data.num_classes,activation="softmax")
])

basic_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

start_time = time.time()
basic_history = basic_model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)
basic_time = time.time() - start_time

basic_acc = max(basic_history.history['val_accuracy'])

#HYBRID MODEL
print("\n🟢 Training Hybrid Model (MobileNet V2)...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
output = layers.Dense(train_data.num_classes, activation="softmax")(x)

hybrid_model = models.Model(inputs=base_model.input, outputs=output)

hybrid_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

start_time = time.time()
hybrid_history = hybrid_model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)
hybrid_time = time.time() - start_time

hybrid_acc = max(hybrid_history.history['val_accuracy'])

#  RESULTS 
print("\n========== FINAL COMPARISON ==========")
print(f"Hybrid Model Accuracy     : {basic_acc:.4f}")
print(f"Basic CNN Accuracy  : {hybrid_acc:.4f}")
print(f"Hybrid Model Time (sec)   : {basic_time:.2f}")
print(f"Basic CNN Time (sec): {hybrid_time:.2f}")

#  GRAPHS 

models_list = ["Basic CNN", "Hybrid"]

accuracy = [basic_acc, hybrid_acc]
time_taken = [basic_time, hybrid_time]

# Accuracy graph
plt.figure()
plt.bar(models_list, accuracy)
plt.title("Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.savefig("accuracy_comparison.png")

# Time graph
plt.figure()
plt.bar(models_list, time_taken)
plt.title("Training Time Comparison")
plt.xlabel("Model")
plt.ylabel("Time (seconds)")
plt.savefig("time_comparison.png")

print("\n📊 Graphs saved:")
print(" - accuracy_comparison.png")
print(" - time_comparison.png")