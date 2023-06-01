import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json



class CustomDataset(Dataset):
    def __init__(self, json_file, transform=None):
        self.data = []
        self.transform = transform

        with open(json_file) as f:
            data = json.load(f)
            for item in data:
                image_path = item['data']
                label = item['annotations']
                self.data.append((image_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]

        # Load and preprocess the image
        image = image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image, label

# Define any necessary transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create an instance of the dataset
dataset = CustomDataset('labels_json/labels.json', transform=transform)

# Create a DataLoader for batching and shuffling
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# for images, labels in dataloader:
#     # Transfer the batch to the GPU if available
#     images = images.to(device='cpu')
#     labels = labels.to(device='cpu')

#     # Forward pass through the model
#     outputs = model(images)

#     # Compute the loss
#     loss = criterion(outputs, labels)

#     # Perform backpropagation and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Print the current loss
#     print('Batch Loss:', loss.item())
