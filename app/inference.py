import cv2
import numpy as np
from tensorflow.keras.models import load_model  # or from torch if using PyTorch

# Load your pre-trained model (replace with your model path)
model = load_model('path_to_your_model.h5')  # For TensorFlow/Keras
# model = torch.load('path_to_your_model.pth')  # For PyTorch (modify accordingly)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-process the frame (resize, convert to grayscale, normalize, etc.)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))  # Resize to the input shape of your model
    normalized = resized / 255.0
    input_image = normalized.reshape(1, 28, 28, 1)  # Adjust shape as required

    # Run inference
    predictions = model.predict(input_image)  # For TensorFlow/Keras
    # predictions = model(input_image)  # For PyTorch (make sure input is a torch tensor)

    # Get predicted number
    predicted_number = np.argmax(predictions)

    # Display the result on the frame
    cv2.putText(frame, f"Predicted: {predicted_number}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Number Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
