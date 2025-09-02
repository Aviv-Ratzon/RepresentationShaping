import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random

# ----------------------------
# 1. Custom dataset
# ----------------------------
class DigitOffsetDataset(Dataset):
    def __init__(self, mnist_dataset, max_offset=3):
        """
        mnist_dataset: torchvision.datasets.MNIST
        max_offset: offsets sampled uniformly from [-max_offset, ..., max_offset], excluding 0
        """
        self.mnist = mnist_dataset
        self.max_offset = max_offset

        # Precompute indices for each digit
        self.class_to_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(self.mnist):
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]  # input digit

        # choose nonzero offset
        offset = random.randint(-self.max_offset, self.max_offset)
        while offset == 0:
            offset = random.randint(-self.max_offset, self.max_offset)

        target_digit = (label + offset) % 10

        # pick random image of target digit
        target_idx = random.choice(self.class_to_indices[target_digit])
        target_img, _ = self.mnist[target_idx]

        # one-hot encode offset in range [-max_offset, max_offset]
        offset_id = offset + self.max_offset  # shift to [0 .. 2*max_offset]
        offset_onehot = F.one_hot(torch.tensor(offset_id), num_classes=2*self.max_offset+1).float()

        return img.view(-1), offset_onehot, target_img.view(-1), label, target_digit


# ----------------------------
# 2. Encoder–Decoder model
# ----------------------------
class DigitTranslator(nn.Module):
    def __init__(self, latent_dim=64, offset_dim=7):  # offset_dim = 2*max_offset+1
        super().__init__()
        self.fc1 = nn.Linear(28*28 + offset_dim, 256)
        self.fc2 = nn.Linear(256, latent_dim)
        self.fc3 = nn.Linear(latent_dim + offset_dim, 256)
        self.fc4 = nn.Linear(256, 28*28)

    def forward(self, x, offset_onehot):
        h = torch.relu(self.fc1(torch.cat([x, offset_onehot], dim=1)))
        z = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(torch.cat([z, offset_onehot], dim=1)))
        x_logits = self.fc4(h)
        return x_logits


# ----------------------------
# 3. Training loop
# ----------------------------
def train_model(epochs=5, batch_size=128, lr=1e-3, max_offset=3, device='cuda'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.view(-1))  # flatten
    ])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    train_ds = DigitOffsetDataset(mnist_train, max_offset=max_offset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = DigitTranslator(offset_dim=2*max_offset+1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, offset, target, _, _ in train_loader:
            x, offset, target = x.to(device), offset.to(device), target.to(device)
            logits = model(x, offset)
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader.dataset):.4f}")

    return model


# ----------------------------
# 4. Sampling
# ----------------------------
def sample(model, input_img, offset, device='cuda'):
    model.eval()
    with torch.no_grad():
        x = input_img.view(1, -1).to(device)
        offset_onehot = F.one_hot(torch.tensor([offset+3]), num_classes=7).float().to(device)
        logits = model(x, offset_onehot)
        img_out = torch.sigmoid(logits).view(28, 28).cpu()
        return img_out


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(epochs=5, device=device)

    # Take an MNIST test image "3" and offset +2 -> "5"
    test_ds = datasets.MNIST(root="./data", train=False, download=True,
                             transform=transforms.ToTensor())
    img, label = test_ds[0]  # first test image
    print("Input digit:", label)

    out_img = sample(model, img, offset=+2, device=device)

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.title(f"Input {label}")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title(f"Output {(label+2)%10}")
    plt.imshow(out_img, cmap="gray")
    plt.show()
