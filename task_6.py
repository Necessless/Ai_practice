from torchvision import transforms
from basics.datasets import CustomImageDataset
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
from basics.utils import make_and_save_graph
import torch

def prepare_data():
    root_train = './data/train'
    root_test = './data/test'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = CustomImageDataset(root_train, transform=transform, target_size=(224, 224))
    test_dataset = CustomImageDataset(root_test, transform=transform, target_size=(224, 224))

    return train_dataset, test_dataset


def train_model(model, optimizer, loss_fn, train_loader, val_loader, device, epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct / total)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_correct / val_total)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train loss: {train_losses[-1]}, acc: {train_accuracies[-1]} - "
              f"Val loss: {val_losses[-1]}, acc: {val_accuracies[-1]}")

    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == "__main__":
    train_data, test_data = prepare_data()
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=32)
    model = efficientnet_b0(pretrained=True).to('cuda')
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(train_data.get_class_names()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, optimizer, loss_fn, train_loader, val_loader, 'cuda', epochs=5
    )
    make_and_save_graph(train_losses, 'results/task_6/train_loss_accuracy_model.png', train_accs)
    make_and_save_graph(val_losses, 'results/task_6/val_loss_accuracy_model.png', val_accs)