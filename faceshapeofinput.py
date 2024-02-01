import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
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
model.load_state_dict(torch.load('C:/Users/Admin/Downloads/best_model.pth', map_location=device))
model.eval()

# Define image preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_image = transform(image).unsqueeze(0)  # Add batch dimension
    return input_image

# Make predictions
input_image_path = 'C:/Users/Admin/Downloads/testimg8.jpg'  # Replace this with the path to your input image
input_image = preprocess_image(input_image_path)
input_image = input_image.to(device)

with torch.no_grad():
    output = model(input_image)

_, predicted_class = torch.max(output, 1)

# Convert the predicted class index to the label
class_idx_to_label = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}
predicted_label = class_idx_to_label[predicted_class.item()]

# Display the input image along with the predicted label
plt.imshow(input_image.squeeze().cpu().permute(1, 2, 0).numpy())
plt.title(f'Predicted Class: {predicted_label}')
plt.show()