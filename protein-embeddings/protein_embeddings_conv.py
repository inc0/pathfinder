# %%
import glob
import numpy as np
import torch
from torch import nn
import torch.optim as optim

from tqdm import tqdm
from torchvision.datasets import DatasetFolder
from torchvision import transforms
# %%


def ds_loader(f):
    data = np.load(f, allow_pickle=True)
    size = data.shape[0]
    data = torch.from_numpy(data.reshape([5, size, size])).type(torch.float32)
    return data


max_size = 1000 
transforms = transform = transforms.Compose([
    transforms.Resize(max_size),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225, 0.225])
])

 # TODO: use Rust CLI to get this number
ds = DatasetFolder(root="/datasets/bigbind/raw_np", loader=ds_loader, extensions="npy", transform=transforms)
loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)

# %%


in_channels = 5
hidden_size = 2048
embedding_size = 1024


class ConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(5,  32, 3, stride=2, padding=0), 
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, stride=2),
            torch.nn.Conv2d(32, 16, 3, stride=2, padding=0), 
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, stride=2),
        )

        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=3, padding=1, output_padding=1),

            nn.ConvTranspose2d(16, 5, kernel_size=4, stride=3, padding=1, output_padding=0),
            nn.BatchNorm2d(5),
            nn.Sigmoid()
        )

    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)
        return decoded


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available
# device = "cpu"
model = ConvAutoencoder().to(device)

from torchsummary import summary
# print(summary(model, (5, 1000, 1000)))

criterion = nn.MSELoss()  # loss function
optimizer = optim.Adam(model.parameters())  # optimizer

# %%
num_epochs = 50  # number of epochs to train for
for epoch in range(num_epochs):
    for i, (inputs, _) in tqdm(enumerate(loader), total=len(loader)):  # no need for labels in autoencoder
        inputs = inputs.to(device)  # move inputs to GPU if available

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)


        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if (i+1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(loader)}], Loss: {loss.item()}')

print('Finished Training')
# %%