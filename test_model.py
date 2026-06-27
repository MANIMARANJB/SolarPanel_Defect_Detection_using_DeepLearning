import tensorflow as tf


print("Loading model...")


model = tf.keras.models.load_model(
    "SolarGuard_MobileNetV2.keras"
)


print("MODEL LOADED SUCCESSFULLY")


model.summary()