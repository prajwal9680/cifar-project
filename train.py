import torch
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import confusion_matrix

from models.resnet import CIFARResNet, BasicBlock

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k * 100 / batch_size)
    return results

def mixup_data(x, y, alpha=1.0 ):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def main():
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                (0.2675, 0.2565, 0.2761))])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565,0.2761))])


    trainset = torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform = train_transform)
    testset = torchvision.datasets.CIFAR100(root='./data', download=True, transform = test_transform, train = False)

    trainloader = DataLoader(trainset, shuffle=True, batch_size=128, num_workers=2, pin_memory = True, persistent_workers = True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    model = CIFARResNet(BasicBlock, [2,2,2,2], num_classes=100)
    model = model.to(device)
    ema_model = CIFARResNet(BasicBlock, [2,2,2,2], num_classes=100).to(device)
    ema_model.load_state_dict(model.state_dict())


    



    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    epochs = 200

    best_acc = 0.0

    for epoch in range(epochs):

        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for inputs, targets in trainloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

            optimizer.zero_grad() 

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model_state = model.state_dict()
                ema_state = ema_model.state_dict()

                for key in ema_state.keys():
                    if ema_state[key].dtype.is_floating_point:
                        ema_state[key].mul_(0.999).add_(0.001 * model_state[key])
                    
                    else:
                        ema_state[key] = model_state[key]

                    
                ema_model.load_state_dict(ema_state)


            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            #correct += (preds == targets).sum().item()
            total += targets.size(0)

        #train_accuracy = 100. * correct / total
        ema_model.eval()
        model.eval()
        correct = 0
        total = 0
        top1_total = 0.0
        top5_total = 0.0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = ema_model(inputs)

                top1, top5 = accuracy(outputs, targets, topk=(1, 5))
                batch_size = targets.size(0)


                
                top1_total += top1.item() * batch_size / 100
                top5_total += top5.item() * batch_size / 100
                total += targets.size(0)
            
        top1_acc = 100. * top1_total / total
        top5_acc = 100. * top5_total / total
        if top1_acc > best_acc:
            best_acc = top1_acc
            torch.save({ "model_state_dict": model.state_dict(),
                          "ema_model_state_dict" : ema_model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                           "epoch":epoch,
                           "best_acc" : best_acc}, 'best_model.pth')

            print(f"new best_acc saved {best_acc:.2f}% ")



        scheduler.step()
        print(f"Epoch: [{epoch+1}/{epochs}] |"
                #f"train_acc: {train_accuracy:.2f}| "
                f"top1_acc: {top1_acc:.2f} |"
                f"top5_acc: {top5_acc:.2f}")

    ema_model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = ema_model(inputs)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)

    class_acc = cm.diagonal() / cm.sum(axis = 1)

    print(f"avg class accuracy is {class_acc.mean() * 100}" )

    best = np.argsort(class_acc)[-5:]
    worst = np.argsort(class_acc)[:5]

    print(f"5 best classes")
    for i in best:
        print(i, class_acc[i] * 100)

    print(f"5 worst classes")
    for i in worst:
        print(i, class_acc[i] * 100)



if __name__ == '__main__':
    main()
