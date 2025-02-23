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
    def __init__(self, images, actions, positions):
        self.images = images.float() / 255.0  # Normalizing images
        self.actions = actions.float().unsqueeze(1)  
        self.positions = positions.float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.actions[idx], self.positions[idx]

# MLP Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * 128 * 128 + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )


    def forward(self, image, action):
        x = image.view(image.size(0), -1) 
        x = torch.cat((x, action), dim=1)
        return self.fc(x)

def train():
    model = MLP()
    actions = torch.load("./training_data/actions.pt")  # Shape: [1000]
    images = torch.load("./training_data/imgs.pt")  # Shape: [1000, 3, 128, 128]
    positions = torch.load("./training_data/position.pt")  # Shape: [1000, 2]

    actions_test = torch.load("./test_data/actions.pt")  # Shape: [200]
    images_test = torch.load("./test_data/imgs.pt")  # Shape: [200, 3, 128, 128]
    positions_test = torch.load("./test_data/position.pt")  # Shape: [200, 2]

    # Train and test datasets
    train_dataset = ObjectPositionDataset(images, actions, positions)
    test_dataset = ObjectPositionDataset(images_test, actions_test, positions_test)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    lr = 0.00001
    epochs = 200

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []

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

        train_losses.append(total_loss / len(train_loader))

        # Evaluating on test set
        model.eval()
        all_preds, all_targets = [], []
        test_loss = 0

        with torch.no_grad():
            for images, actions, targets in test_loader:
                predictions = model(images, actions)
                loss = criterion(predictions, targets)
                test_loss += loss.item()

                all_preds.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        test_losses.append(test_loss / len(test_loader))

        # Converting lists to NumPy arrays
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        # Computing Metrics
        mae = np.mean(np.abs(all_targets - all_preds))  # MAE
        rmse = np.sqrt(np.mean((all_targets - all_preds) ** 2))  # RMSE
        
        # Computing R² Score
        total_variance = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2)
        unexplained_variance = np.sum((all_targets - all_preds) ** 2)
        r2 = 1 - (unexplained_variance / total_variance) if total_variance != 0 else float('nan')

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_losses[-1]:.4f}")
        print(f"  Test Loss: {test_losses[-1]:.4f}")
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")
        print("-" * 40)

    torch.save(model.state_dict(), "hw1_1.pt")
    print("Model saved as hw1_1.pt")

    # Plotting loss
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="MLP Train Loss", marker="o")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="MLP Test Loss", marker="o")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("MLP Training vs. Test Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def test():
    model = MLP()
    model.load_state_dict(torch.load("hw1_1.pt"))
    model.eval()

    actions_test = torch.load("./test_data/actions.pt")  
    images_test = torch.load("./test_data/imgs.pt")  
    positions_test = torch.load("./test_data/position.pt")  

    test_dataset = ObjectPositionDataset(images_test, actions_test, positions_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    test_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, actions, targets in test_loader:
            predictions = model(images, actions)
            loss = criterion(predictions, targets)
            test_loss += loss.item()

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    test_loss /= len(test_loader)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    mae = np.mean(np.abs(all_targets - all_preds))
    rmse = np.sqrt(np.mean((all_targets - all_preds) ** 2))
    
    total_variance = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2)
    unexplained_variance = np.sum((all_targets - all_preds) ** 2)
    r2 = 1 - (unexplained_variance / total_variance) if total_variance != 0 else float('nan')

    print(f"Test Loss: {test_loss:.4f}")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")

if __name__ == "__main__":
    test()