import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


from utils import set_seed
from models.baseline import BaselineCNN



def compute_accuracy(outputs, labels):
    
    _,preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device used is {device}")

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding = 4), transforms.ToTensor()])
    test_transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(root = "./data", train = True, download = True, transform = train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root = './data', download = True, train = False, transform =test_transform )

    batch_size = 128
    train_loader = DataLoader(train_dataset, shuffle = True, pin_memory = torch.cuda.is_available(), batch_size = batch_size, num_workers = 2)

    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 2, pin_memory = torch.cuda.is_available())

    

    epochs = 10

    model = BaselineCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum = 0.9, lr = 0.01, weight_decay = 5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
    

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += compute_accuracy(outputs, labels)
            train_total += labels.size(0)
        train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += compute_accuracy(outputs, labels)
                val_total += labels.size(0)
            val_loss = val_loss / val_total
            val_accuracy = val_correct / val_total
        print(f"at [{epoch + 1}/{epochs}] |"
                f"Learning_rate {current_lr:.4f}|"
                f"train_loss {train_loss:.4f}|"
                f"train_accuracy {train_accuracy:.4f} |"
                f"val_loss {val_loss:.4f} |"
                f"val_accuracy {val_accuracy:.4f} |")

        scheduler.step()
    
    from evaluate import evaluate_model, compute_confusion_matrix, compute_per_class_accuracy, plot_confusion_matrix

    class_names = train_dataset.classes
    preds, labels = evaluate_model(model, test_loader, device)

    conf_matrix = compute_confusion_matrix(preds, labels)
    per_class_acc = compute_per_class_accuracy(conf_matrix)

    print("Per class accuracy")
    for i, acc in enumerate(per_class_acc):
        print(f"{class_names[i] } : {acc : .4f}")

    plot_confusion_matrix(conf_matrix, class_names)


if __name__ == "__main__":
    main()