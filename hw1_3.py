import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
import subprocess
import sys

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt 

warnings.simplefilter("ignore", category=FutureWarning)

# Dataset class
class ObjectPositionDataset(Dataset):
    def __init__(self, images_before, actions, images_after):
        self.images_before = images_before.float() / 255.0  # Normalizing images
        self.actions = actions.float().unsqueeze(1)  # Ensuring actions are 2D
        self.images_after = images_after.float() / 255.0

    def __len__(self):
        return len(self.images_before)

    def __getitem__(self, idx):
        return self.images_before[idx], self.actions[idx], self.images_after[idx]

# CNN Autoencoder Model
class CNN_Autoencoder(nn.Module):
    def __init__(self):
        super(CNN_Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8 + 1, 2048),  # Change input size here to match the concatenated tensor
            nn.ReLU(),
            nn.Linear(2048, 128 * 8 * 8),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, image, action):
        x = self.encoder(image)
        x = torch.cat((x, action), dim=1)
        x = self.fc(x)
        x = self.decoder(x)
        return x


def train():
    model = CNN_Autoencoder()
    actions = torch.load("./training_data_part3/actions.pt")
    images_before = torch.load("./training_data_part3/imgs_before.pt")
    images_after = torch.load("./training_data_part3/imgs_after.pt")

    actions_test = torch.load("./test_data_part3/actions.pt")  
    images_before_test = torch.load("./test_data_part3/imgs_before.pt")  
    images_after_test = torch.load("./test_data_part3/imgs_after.pt")  

    train_dataset = ObjectPositionDataset(images_before, actions, images_after)
    test_dataset = ObjectPositionDataset(images_before_test, actions_test, images_after_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    epochs = 100

    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, actions, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(images, actions)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for images, actions, targets in test_loader:
                predictions = model(images, actions)
                loss = criterion(predictions, targets)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    
    torch.save(model.state_dict(), "hw1_3.pt")
    print("Model saved as hw1_3.pt")
    
    # Plot training and test loss
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss", marker="o", linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

def test():
    model = CNN_Autoencoder()
    model.load_state_dict(torch.load("hw1_3.pt"))
    model.eval()

    actions_test = torch.load("./test_data_part3/actions.pt")  
    images_before_test = torch.load("./test_data_part3/imgs_before.pt")  
    images_after_test = torch.load("./test_data_part3/imgs_after.pt")  

    test_dataset = ObjectPositionDataset(images_before_test, actions_test, images_after_test)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    criterion = nn.MSELoss()
    test_loss = 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, actions, next_images in test_loader:
            predictions = model(images, actions)
            loss = criterion(predictions, next_images)
            test_loss += loss.item()

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(next_images.cpu().numpy())
    
    test_loss /= len(test_loader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    print(f"Test Loss: {test_loss:.4f}")
    
    # Visualizing some predictions
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))  # 3 rows: original, target, predicted
    for i in range(8):
        axes[0, i].imshow(images_before_test[i].permute(1, 2, 0).numpy())  # Original image
        axes[0, i].axis("off")
        
        axes[1, i].imshow(all_targets[i].transpose(1, 2, 0))  # Ground-truth next image
        axes[1, i].axis("off")
        
        axes[2, i].imshow(all_preds[i].transpose(1, 2, 0))  # Predicted next image
        axes[2, i].axis("off")
    
    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Target")
    axes[2, 0].set_title("Predicted")
    plt.show()

if __name__ == "__main__":
    test()
