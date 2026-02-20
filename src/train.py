import torch
import torch.optim as optim
from tqdm import tqdm
from model import build_model
from dataset import get_dataloaders

def train_model(dataset_base, model_save_path, device):

    train_loader, val_loader, class_names = get_dataloaders(dataset_base)
    model = build_model(len(class_names), device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    best_val_acc = 0

    for epoch in range(25):

        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs,1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs,1)

                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}")
        print("Train Acc:", train_acc)
        print("Val Acc:", val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

    print("Best Val Accuracy:", best_val_acc)