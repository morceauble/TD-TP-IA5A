import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the batch size and the number of epochs
batch_size = 128
epochs = 15

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the denoising autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28x1 to 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28x32 to 14x14x32
            nn.Conv2d(32, 32, 3, padding=1),  # 14x14x32 to 14x14x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 14x14x32 to 7x7x32
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),  # 7x7x32 to 7x7x32
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 7x7x32 to 14x14x32
            nn.Conv2d(32, 32, 3, padding=1),  # 14x14x32 to 14x14x32
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 14x14x32 to 28x28x32
            nn.Conv2d(32, 1, 3, padding=1),  # 28x28x32 to 28x28x1
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Lists to store training and testing loss and accuracy
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for data in train_loader:
        inputs, _ = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Correctly predicted pixels
        correct_pixels = (torch.abs(outputs - inputs) < 0.5).sum()
        total_pixels = inputs.numel()
        correct_train += correct_pixels.item()
        total_train += total_pixels

    train_loss_values.append(running_loss / len(train_loader))
    train_accuracy = 100 * correct_train / total_train
    train_accuracy_values.append(train_accuracy)

    # Testing loop
    model.eval()
    running_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_loss += loss.item()

            # Correctly predicted pixels
            correct_pixels = (torch.abs(outputs - inputs) < 0.5).sum()
            total_pixels = inputs.numel()
            correct_test += correct_pixels.item()
            total_test += total_pixels

    test_loss_values.append(running_loss / len(test_loader))
    test_accuracy = 100 * correct_test / total_test
    test_accuracy_values.append(test_accuracy)

# Plot training and testing accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_accuracy_values, label='Train Accuracy')
plt.plot(range(1, epochs + 1), test_accuracy_values, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Testing Accuracy')
plt.show()

# Plot training and testing loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_loss_values, label='Train Loss')
plt.plot(range(1, epochs + 1), test_loss_values, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss')
plt.show()

# Visualization of original, noisy, and reconstructed images
n = 10
noise_factor = 0.5

plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    img, _ = test_dataset[i]
    plt.imshow(img.squeeze().numpy(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + n)
    img_noisy = img + noise_factor * torch.randn(1, 28, 28)
    img_noisy = torch.clamp(img_noisy, 0., 1.)
    plt.imshow(img_noisy.squeeze().numpy(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    reconstructed_img = model(img_noisy.unsqueeze(0))
    plt.imshow(reconstructed_img.squeeze().detach().numpy(), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()