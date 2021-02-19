from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
from net import Net as CNN
from load_data import ImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Custom Network
net = CNN().to(device)
summary(net, (3, 224, 224))
#print(net)

# Define Data paths
train_path = 'dataset/set_dataset/train'
train_labels = 'dataset/set_dataset/train_labes.csv'
val_path = 'dataset/set_dataset/val'
val_labels = 'dataset/set_dataset/val_labes.csv'
test_path = 'dataset/set_dataset/test'
test_labels = 'dataset/set_dataset/test_labes.csv'

# TODO: Complete the Normalization function
def mean_std_calc(data):
    pass

mean_val = [0.5, 0.5, 0.5]
std_val = [0.5, 0.5, 0.5]
model = CNN().to(device)


train_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean_val, std_val),
])

test_val_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean_val, std_val),
])

train_data = ImageDataset(train_labels, train_path, train_transforms)
val_data = ImageDataset(val_labels, val_path, test_val_transforms)
test_data = ImageDataset(test_labels, test_path, test_val_transforms)

train_loader = DataLoader(train_data, batch_size=6, shuffle=True, num_workers=4)
valloader = DataLoader(val_data, batch_size=6, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=6, shuffle=True, num_workers=4)

classes = ('bolt', 'flange', 'lead_block', 'nut', 'pipe')

train_size = len(train_data)
val_size = len(val_data)
test_size = len(test_data)

print(f"Train data size: {train_size}\nValidation data size: {val_size}\n\
Test data size: {test_size}\nTotal data size: {train_size + val_size + test_size}")


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epochs = 2

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 20 == 0:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
            
print('Finish Training')

# save the model
PATH = 'models/my_simple_net.pth'
torch.save(net.state_dict(), PATH)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {test_size} test images: %d %%' % (
    100 * correct / total))