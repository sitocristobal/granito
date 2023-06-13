import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

# Función para cargar el archivo JSON y procesar los datos
def load_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    
    images = []
    labels = []
    
    for item in data:
        image_path = item["image"]
        label = item["tag"]
        
        image = Image.open(image_path)
        image = image.resize((128, 128))
        image = ToTensor()(image)
        
        images.append(image)
        labels.append(label)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return images, labels

# Paso 1: Cargar los datos desde el archivo JSON
train_images, train_labels = load_json('labels_json\est3json.json')

# Paso 2: Dividir los datos en conjuntos de entrenamiento y prueba
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Paso 3: Definir el modelo de CNN utilizando PyTorch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# Paso 4: Crear una instancia del modelo de CNN
model = CNN()

# Paso 5: Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Paso 6: Entrenar el modelo
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i in range(0, len(train_images), batch_size):
        inputs = train_images[i:i+batch_size]
        labels = train_labels[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_images)}")

# Paso 7: Evaluar el modelo
model.eval()
with torch.no_grad():
    outputs = model(test_images)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == test_labels).sum().item() / len(test_labels)
    print(f"Accuracy: {accuracy * 100}%")
