# ======================
# STEP 1: Load and Preprocess Data
# ======================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocessing
X_train = X_train.reshape(60000, 784).astype('float32') / 255  # Flatten + normalize
X_test = X_test.reshape(10000, 784).astype('float32') / 255
y_train = to_categorical(y_train, 10)  # One-hot encode labels
y_test = to_categorical(y_test, 10)

# ======================
# STEP 2: Define MLP Model
# ======================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # Hidden layer 1
    Dense(64, activation='relu'),                       # Hidden layer 2
    Dense(10, activation='softmax')                     # Output layer
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ======================
# STEP 3: Train the Model
# ======================
history = model.fit(
    X_train, y_train,
    epochs=15,  # Increased from 10 to 15 for better convergence
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ======================
# STEP 4: Evaluate the Model
# ======================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'\nTest accuracy: {test_acc*100:.2f}%')

# ======================
# STEP 5: Visualization
# ======================
# Plot training history
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ======================
# STEP 6: Sample Predictions
# ======================
# Select 10 random test images
sample_indices = np.random.choice(len(X_test), 10)
sample_images = X_test[sample_indices]
sample_labels = np.argmax(y_test[sample_indices], axis=1)

# Make predictions
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

# Display results
plt.figure(figsize=(15, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'True: {sample_labels[i]}\nPred: {predicted_labels[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()