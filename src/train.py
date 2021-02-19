import os
from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_net import Net
from load_data import ImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
from PIL import Image, ImageFile
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Custom Network
model = Net().to(device)
summary(model, (3, 224, 224))
# print(net)

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
val_loader = DataLoader(val_data, batch_size=6, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=6, shuffle=True, num_workers=4)

classes = ('bolt', 'flange', 'lead_block', 'nut', 'pipe')

train_size = len(train_data)
val_size = len(val_data)
test_size = len(test_data)

print(f"Train data size: {train_size}\nValidation data size: {val_size}\n\
Test data size: {test_size}\nTotal data size: {train_size + val_size + test_size}")

optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, optimizer, loss_func, train_loader, val_loader, epochs=20, device='cpu'):
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item() * inputs.size(0)  # CHECK (is the   # batch size and used to get the loss of a
            # batch when batch size is not a factor of train_size)
        train_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0.0
        num_examples = 0.0
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(outputs, dim=1), dim=1)[1], labels)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]

        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, 
                                train_loss,
                                valid_loss,
                                num_correct / num_examples))


train(model, optimizer, torch.nn.CrossEntropyLoss(), train_loader, val_loader, epochs=5, device=device)

# save the model depending the timestamp
date = time.strftime("%d%m%Y-")
NAME = f"model_{date}{int(time.time())}.pth"
PATH = 'models/'
torch.save(model.state_dict(), os.path.join(PATH, NAME))


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {test_size} test images: %d %%' % (
        100 * correct / total))

img = Image.open('dataset/set_dataset/test/bolt_12.jpg')
img = test_val_transforms(img).to(device)
img = torch.unsqueeze(img, 0)

model.eval()
prediction = F.softmax(model(img), dim=1)
prediction = prediction.argmax()
print(prediction)
print(classes[prediction])


