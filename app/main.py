# KNN's Implementaion Starter Code

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load and preprocess data
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(np.int8)
X /= 255.0  # Normalize

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate model
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize predictions
plt.figure(figsize=(10, 10))
for i in range(16):
    index = np.random.randint(0, len(X_test))
    img = X_test[index].reshape(28, 28)
    plt.subplot(4, 4, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Pred: {y_pred[index]}")
    plt.axis('off')
plt.show()
