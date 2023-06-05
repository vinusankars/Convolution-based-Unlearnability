import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib
from util import AverageMeter
from resnet import resnet18 as net
from six.moves import cPickle as pkl
import torchattacks

train_transform = [transforms.ToTensor()]
test_transform = [transforms.ToTensor()]
train_transform = transforms.Compose(train_transform)
test_transform = transforms.Compose(test_transform)
batch_size = 500

clean_train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)
clean_test_dataset = datasets.CIFAR10(root='../datasets', train=False, download=True, transform=test_transform)

clean_train_loader = DataLoader(dataset=clean_train_dataset, batch_size=batch_size,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=12)
                                
clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=batch_size,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=12)


# unlearnable parameters
grayscale = False # grayscale
blur_parameter = 0.3
center_parameter = 1.0
kernel_size = 3
seed = 0
same = False

# test parameters
test_grayscale = False

def get_filter_unlearnable(blur_parameter, center_parameter, grayscale, kernel_size, seed, same):

    np.random.seed(seed)
    cnns = []
    with torch.no_grad():
        for i in range(10): 
            cnns.append(torch.nn.Conv2d(3, 3, kernel_size, groups=3, padding=1).cuda())
            if blur_parameter is None:
                blur_parameter = 1
            w = np.random.uniform(low=0, high=blur_parameter, size=(3,1,3,3))
            # w = np.random.random((3,1,3,3))
            if center_parameter is not None:
                shape = w[0][0].shape
                w[0, 0, np.random.randint(shape[0]), np.random.randint(shape[1])] = center_parameter
            w[1] = w[0]
            w[2] = w[0]
            # w = w/w.max()
            cnns[i].weight.copy_(torch.tensor(w))
            cnns[i].bias.copy_(cnns[i].bias * 0)
    cnns = np.stack(cnns)

    if same:
        cnns = np.stack([cnns[0]] * len(cnns))

    unlearnable_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)
    unlearnable_loader = DataLoader(dataset=unlearnable_dataset, batch_size=500,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False, num_workers=12)

    pbar = tqdm(unlearnable_loader, total=len(unlearnable_loader))
    images_ = []

    for images, labels in pbar:
        images, labels = images.cuda(), labels.cuda()
        for i in range(len(images)):
            id = labels[i].item()
            img = cnns[id](images[i:i+1]).detach().cpu() # convolve class-wise
            # # black and white
            if grayscale:
                img_bw = img[0].mean(0)
                img[0][0] = img_bw
                img[0][1] = img_bw
                img[0][2] = img_bw        
            images_.append(img/img.max())

    # making unlearnable data
    unlearnable_dataset.data = unlearnable_dataset.data.astype(np.float32)
    for i in range(len(unlearnable_dataset)):
        unlearnable_dataset.data[i] = images_[i][0].numpy().transpose((1,2,0))*255
        unlearnable_dataset.data[i] = np.clip(unlearnable_dataset.data[i], a_min=0, a_max=255)
    unlearnable_dataset.data = unlearnable_dataset.data.astype(np.uint8)

    return unlearnable_dataset, cnns


unlearnable_dataset, cnns = get_filter_unlearnable(blur_parameter, center_parameter, grayscale, kernel_size, seed, same)


# get ready for training
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

clean_train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)
unlearnable_dataset.transforms = clean_train_dataset.transforms
unlearnable_dataset.transform = clean_train_dataset.transform

unlearnable_loader = DataLoader(dataset=unlearnable_dataset, batch_size=batch_size,
                                shuffle=True, pin_memory=True,
                                drop_last=False, num_workers=12)

clean_train_loader = DataLoader(dataset=clean_train_dataset, batch_size=batch_size,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=12)

clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=batch_size,
                                shuffle=True, pin_memory=True,
                                drop_last=False, num_workers=12)

# Below is the FAT attack procedure
class Attack():

    def __init__(self, steps, eta, criterion):
        self.steps = steps 
        self.eta = eta 
        self.criterion = criterion

    def model_gradients(self, model, value=True):
        for param in model.parameters():
            param.requires_grad = value

    def apply_constraints(self):

        for i in range(len(self.tcnns)):
            with torch.no_grad():
                temp = self.tcnns[i].weight
                # shape = temp[0, 0].shape
                M = 5
                # temp[0, 0, shape[0]//2, shape[1]//2] = M
                self.tcnns[i].weight.copy_(torch.clamp(temp, -M, M))
                temp = self.tcnns[i].bias
                self.tcnns[i].bias.copy_(torch.clamp(temp, -M, M))

    def perturb(self, model, x, y):

        self.model_gradients(model, False)
        filter_size = 7
        num_classes = 10
        self.tcnns = [torch.nn.ConvTranspose2d(1, 1, filter_size, groups=1, padding=filter_size//2).cuda() for i in range(num_classes)]

        for step in range(self.steps):

            self.apply_constraints()
            opt = torch.optim.SGD([i.weight for i in self.tcnns] + [i.bias for i in self.tcnns], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()

            for cls in range(10):

                idx = (labels == cls)
                x_, y_ = self.tcnns[cls](x[idx].view(len(x[idx])*3, 1, 32, 32)).view(len(x[idx]), 3, 32, 32), y[idx]
                logits = model(x_)

                loss = self.criterion(logits, y_)
                self.tcnns[cls].weight.retain_grad()
                self.tcnns[cls].bias.retain_grad()
                loss.backward()
                
                self.tcnns[cls].weight.data += self.tcnns[cls].weight.grad.data * (+1) * self.eta
                self.tcnns[cls].bias.data += self.tcnns[cls].bias.grad.data * (+1) * self.eta

        x_ = torch.zeros(x.shape)
        for cls in range(10):
            idx = (labels == cls)
            xs = self.tcnns[cls](x[idx].view(len(x[idx])*3, 1, 32, 32)).view(len(x[idx]), 3, 32, 32)
            x_[idx] = xs.clone().detach().cpu()

        for i in range(len(x_)):
            x_[i] -= x_[i].min() 
            x_[i] /= x_[i].max()

        self.model_gradients(model, True)

        return x_


# multiplicative adversarial training / Filter Adversarial Training
epochs = 100
model = net(3, 10)
model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(epochs*0.4), gamma=0.1)
# atk = torchattacks.PGD(model, eps=8/255, alpha=8/2550, steps=10)
atk = Attack(steps=10, eta=0.1, criterion=criterion)

train_acc = []
test_acc = []

try:
    for epoch in range(epochs):
        # Train
        model.train()
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(unlearnable_loader, total=len(unlearnable_loader))

        for images, labels in pbar:
            images, labels = images.cuda(), labels.cuda()
            model.zero_grad()
            optimizer.zero_grad()
            # adv_images = atk(images, labels)
            # logits = model(adv_images)
            adv_images = atk.perturb(model, images, labels)
            logits = model(adv_images.cuda())
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            train_acc.append(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))

        scheduler.step()

        # Eval
        model.eval()
        correct, total = 0, 0

        for i, (images, labels) in enumerate(clean_test_loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        acc = correct / total
        test_acc.append(acc)
        tqdm.write('Clean Accuracy %.2f\n' % (acc*100))
        tqdm.write('Epoch %.2f\n' % (epoch))

except:

    bp = (blur_parameter is None)*'x' + str(blur_parameter)
    cp = (center_parameter is None)*'x' + str(center_parameter)
    kp = str(kernel_size)
    sp = str(seed)
    gp = str(int(grayscale))
    tgp = str(int(test_grayscale))

    with open('results/muladv_train_size7.pkl'.format(bp, cp, kp, sp, gp, tgp, int(same)), 'wb') as f:
        pkl.dump([train_acc, test_acc], f)

bp = (blur_parameter is None)*'x' + str(blur_parameter)
cp = (center_parameter is None)*'x' + str(center_parameter)
kp = str(kernel_size)
sp = str(seed)
gp = str(int(grayscale))
tgp = str(int(test_grayscale))

with open('results/muladv_train_size7.pkl'.format(bp, cp, kp, sp, gp, tgp, int(same)), 'wb') as f:
    pkl.dump([train_acc, test_acc], f)