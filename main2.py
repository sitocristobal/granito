import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

# Paso 1: Procesamiento de datos
img_folder = 'pictures_test/RP'
label_folder = 'labels_test/RP'

# Paso 2: Preparación del entorno

# Verificar soporte de GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paso 3: Carga de datos
class ImageDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None, num_classes=1):
        # Inicialización del conjunto de datos
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = os.listdir(image_folder)
        self.label_files = os.listdir(label_folder)
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        # Obtener nombres de archivos de imagen y etiqueta
        image_filename = self.image_files[index]
        label_filename = self.label_files[index]
        image_path = os.path.join(self.image_folder, image_filename)
        label_path = os.path.join(self.label_folder, label_filename)

        # Abrir imágenes y etiquetas
        image = Image.open(image_path)
        label = Image.open(label_path)

        # Aplicar transformaciones a las imágenes si se proporciona un transformador
        if self.transform is not None:
            image = self.transform(image)

        # Convertir etiqueta a tensor y aplicar codificación one-hot
        num_classes = 1
        label_tensor = torch.tensor([1.0 if i == int(label_filename.split('.')[0]) else 0.0 for i in range(num_classes)], dtype=torch.float32)

        return image, label_tensor

# Transformaciones de datos para normalizar y redimensionar imágenes
transform = transforms.Compose((
    transforms.Resize((224, 224)),  # Redimensiona las imágenes a un tamaño específico
    transforms.ToTensor(),  # Convierte las imágenes a tensores
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza los tensores de imágenes
))

# Carga de imágenes de laboratorio de granito y segmentaciones
dataset = ImageDataset(img_folder, label_folder, transform=transform, num_classes=1)

# División de datos en conjuntos de entrenamiento, validación y prueba
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Paso 4: Definición del modelo
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1)  # Solo una salida para clasificación binaria
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = CNN(num_classes=1)
model = model.to(device)

# Paso 5: Entrenamiento del modelo

# Definir función de pérdida y optimizador
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Definir los cargadores de datos para entrenamiento y validación
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# Entrenamiento del modelo
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Realizar pasos hacia adelante y hacia atrás
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validar el modelo después de cada época
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    
    accuracy = total_correct / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy}")
