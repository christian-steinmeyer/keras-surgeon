from keras.src.applications import inception_v3
from keras.src.preprocessing.image import ImageDataGenerator

from kerassurgeon.examples.inception_flowers_tune import (
    batch_size,
    img_height,
    img_width,
    train_data_dir,
    validation_data_dir,
)


def get_data_generators():
    # Prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        preprocessing_function=inception_v3.preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
    )
    test_datagen = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input)
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
    )
    return train_generator, validation_generator
