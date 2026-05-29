from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from config import IMG_SIZE, LEARNING_RATE

def build_model(num_classes):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
