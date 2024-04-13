# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:08:22 2024

@author: mskam
"""
# Import necessary libraries
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import os

# Define functions for model training, loading, and prediction
    
def train_model2():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Data preprocessing
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Define the model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model with data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='nearest'
    )
    
    model.fit(datagen.flow(x_train, y_train, batch_size=32),
              epochs=10,
              validation_data=(x_test, y_test))
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    
    # Save the model
    model.save('mnist_model.keras')

def load_model():
    # Load the saved model
    loaded_model = tf.keras.models.load_model('mnist_model.keras')
    return loaded_model

def preprocess_image(image, model):
    # Convert the image to RGBA mode
    image_rgba = image.convert("RGBA")
    
    # Extract the alpha channel (4th layer)
    alpha_channel = image_rgba.split()[-1]
    
    # Convert the alpha channel to a numpy array
    alpha_array = np.array(alpha_channel)
    
    # Resize the alpha channel image
    resized_alpha = Image.fromarray(alpha_array).resize((28, 28))
    
    # Convert the resized alpha channel to a numpy array
    alpha_array_resized = np.array(resized_alpha)
    
    # Normalize the alpha channel array
    normalized_alpha = alpha_array_resized / 255.0
    
    # Add an extra dimension to match the model's input shape
    preprocessed_alpha = normalized_alpha[..., np.newaxis]

    return preprocessed_alpha

def predict_digit(model, image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image, model)

    # Make prediction using the model
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    
    # Get the predicted digit
    predicted_digit = np.argmax(prediction)
    confidence_score = np.max(prediction)

    # Print prediction details
    print("Predicted Digit:", predicted_digit)
    print("Confidence Score:", confidence_score)

    return predicted_digit, confidence_score
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make prediction using the loaded model
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    
    # Get the predicted digit and confidence score
    predicted_digit = np.argmax(prediction)
    confidence_score = np.max(prediction)

    # Print the prediction and layer information
    print("Predicted Digit:", predicted_digit)
    print("Confidence Score:", confidence_score)
    print("Layer Information:", model.layers)  # Print information about all layers in the model

    return predicted_digit, confidence_score

# Streamlit app
def model_exists():
    return os.path.exists('mnist_model.keras')

def main():
    # Train the model if necessary
    train = st.checkbox("Train the Model", value=False)
    
    # Load the model if it exists and the checkbox is unchecked
    if not train and model_exists():
        model = load_model()
        st.write("The model is available and ready for use")
    else:
        train_model2()
        st.write("Model trained successfully!")
        model = load_model()
        train = False

    # Title and description
    st.title("MNIST Digit Recognition App")
    st.write("Upload an image of a digit (0-9)")

    # Allow user to upload an image
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Process and predict the digit
        pil_image = Image.open(uploaded_image)
        digit, confidence = predict_digit(model, pil_image)
        
        #debug_preprocessing(pil_image)

        # Display the predicted digit and confidence score
        st.write(f"Predicted Digit: {digit} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()
