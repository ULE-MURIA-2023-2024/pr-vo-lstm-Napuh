import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import VisualOdometryDataset
from model import VisualOdometryModel
from params import batch_size, epochs, hidden_size, learning_rate, num_layers, sequence_length, bidirectional, lstm_dropout


# Create the visual odometry model
model = VisualOdometryModel(hidden_size, num_layers, bidirectional, lstm_dropout)

transform = T.Compose([
    T.ToTensor(),
    model.resnet_transforms()
])


# TODO: Load the dataset
train_dataset = VisualOdometryDataset(dataset_path="./dataset/train", 
                                      transform=transform, 
                                      sequence_length=sequence_length,
                                      validation=False)

# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# # train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
running_loss = 0.0

for epoch in range(epochs):

    for images, labels, _ in tqdm(train_loader, f"Epoch {epoch + 1}:"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}")
    running_loss = 0.0

torch.save(model.state_dict(), "./vo.pt")
