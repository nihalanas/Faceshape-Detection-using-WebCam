import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model architecture
model = models.efficientnet_b4(pretrained=False)
num_classes = 5
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(model.classifier[1].in_features, num_classes)
)

# Load the trained weights
model.load_state_dict(torch.load('---pathtothemodel---/best_model.pth', map_location=device))
model.eval()

# Define image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_image = transform(image).unsqueeze(0)  # Add batch dimension
    return input_image

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Face Shape Prediction', frame)

    # Wait for the spacebar key (key code 32)
    if cv2.waitKey(1) & 0xFF == 32:
        # Convert the frame to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to PIL Image
        pil_image = Image.fromarray(rgb_frame)

        # Preprocess the image
        input_image = preprocess_image(pil_image)
        input_image = input_image.to(device)

        with torch.no_grad():
            output = model(input_image)

        _, predicted_class = torch.max(output, 1)

        # Convert the predicted class index to the label
        class_idx_to_label = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}
        predicted_label = class_idx_to_label[predicted_class.item()]

        # Print the predicted label
        print(f'Predicted Class: {predicted_label}')

        # You can save the frame or perform other actions here if needed

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
