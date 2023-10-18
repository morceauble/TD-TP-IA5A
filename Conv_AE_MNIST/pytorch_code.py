import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Define the model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # Add padding to match the size of the input tensor
        x = nn.functional.pad(x, (4, 4, 4, 4), mode='constant', value=0)
        return x

# Define the training function
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = criterion(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        # Calculate accuracy
        pred = recon_batch > 0.5
        acc = (pred == data).sum().item() / (data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3])
        train_acc += acc
        
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader)
    print('Epoch: {} Train loss: {:.4f} Train accuracy: {:.4f}'.format(epoch, train_loss, train_acc))
    return train_loss, train_acc

# Load the data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# Initialize the model, optimizer and loss function
model = Autoencoder()
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

# Train the model
train_losses = []
train_accs = []
test_losses = []
test_accs = []
for epoch in range(1, 51):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Evaluate on test set
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, _ in test_loader:
            recon_batch = model(data)
            loss = criterion(recon_batch, data)
            test_loss += loss.item()
            
            # Calculate accuracy
            pred = recon_batch > 0.5
            acc = (pred == data).sum().item() / (data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3])
            test_acc += acc
            
    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

# Plot the results
epochs = range(1, 51)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train')
plt.plot(epochs, test_losses, label='Test')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, label='Train')
plt.plot(epochs, test_accs, label='Test')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Select 10 images from the test set
dataiter = iter(test_loader)
images, labels = dataiter.next()
images = images[:10]

# Reconstruct the images using the trained model
reconstructions = model(images)

# Plot the original and reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

for i in range(10):
    # Plot the original images
    axes[0][i].imshow(images[i].squeeze(), cmap='gray')
    axes[0][i].set_title('Original')
    axes[0][i].get_xaxis().set_visible(False)
    axes[0][i].get_yaxis().set_visible(False)

    # Plot the reconstructed images
    axes[1][i].imshow(reconstructions[i].detach().squeeze(), cmap='gray')
    axes[1][i].set_title('Reconstructed')
    axes[1][i].get_xaxis().set_visible(False)
    axes[1][i].get_yaxis().set_visible(False)

plt.show()
