import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 1. Load the processed data
print("Loading data...")
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

# 2. Split into Training (80%) and Validation (20%) sets
# random_state ensures we get the same split every time we run the script
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")

# 3. Build the hardware-efficient CNN Architecture
def build_model(input_shape):
    model = models.Sequential()
    
    # First Convolutional Block (Extracts basic audio features)
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second Convolutional Block (Extracts more complex patterns)
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten the 2D matrix into a 1D array for the final decision
    model.add(layers.Flatten())
    
    # Dense layer for reasoning
    model.add(layers.Dense(32, activation='relu'))
    
    # Output layer: 1 node with Sigmoid activation
    # Outputs a probability between 0.0 (Safe) and 1.0 (Chainsaw)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# The shape is (64, 157, 1) based on our previous extraction
input_shape = X_train.shape[1:] 
model = build_model(input_shape)

# 4. Compile the model
# Adam is a fast optimizer, and binary_crossentropy is perfect for our Yes/No classification
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# 5. Train the Model!
print("\nStarting training...")
# Epochs = how many times the AI loops through the entire dataset
# Batch size = how many spectrograms it looks at before updating its math
history = model.fit(X_train, y_train, 
                    epochs=15, 
                    batch_size=16, 
                    validation_data=(X_test, y_test))

# 6. Save the trained model
model.save("chainsaw_detector.keras")
print("\nModel successfully trained and saved as 'chainsaw_detector.keras'!")