import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from PIL import Image

# Step 1: Load and transform the data
def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Loads and processes the data from the JSON file according to its structure
    # Returns images and labels in the proper format
    images = data['image']
    labels = data['tag']
    return images, labels

# Define transformations for image pre-processing
transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize to a fixed size
    transforms.ToTensor(), # convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # normalize
])

# Load the data from JSON file
train_images, train_labels = load_json('labels_json\labels.json')

# Create a custom dataset
train_dataset = DatasetFolder(train_images, train_labels, transform=transform)

num_classes = 1 # Number of defects to detect

# Step 2: Define the model
class CNN(nn.Module):
    def __init__(self):
        # Define the architecture of the CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 112 * 112, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        return x

model = CNN()


# Step 3: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Step 4: Train the model
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 10
device = torch.device('cpu')
model.to(device)
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


# Step 5: Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)


            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total 
    print(f"Accuracy: {accuracy}%")

# Load test data from JSON file
test_images, test_labels = load_json('labels_json\labels.json')

# Create a test data set
test_dataset = DatasetFolder(test_images, test_labels, transform=transform)

# Create the DataLoader for the test data set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model on the test data set
evaluate_model(model, test_loader)


# Step 6: making predictions
def predict_image(model, image):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

# Load an unlabeled image to make the prediction
image_path = 'image.jpg'
unlabeled_image = Image.open(image_path)

# Make the prediction on the unlabeled image
prediction = predict_image(model, unlabeled_image)
print(f"Prediction: {prediction}")

