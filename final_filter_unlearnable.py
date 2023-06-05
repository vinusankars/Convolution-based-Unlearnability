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
# from ResNet import ResNet18
from six.moves import cPickle as pkl
import argparse
import os
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg16-bn', 'densenet-121', 'wrn-34-10'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--train-type', type=str, default='erm', choices=['erm', 'adv'], help='ERM or Adversarial training loss')
parser.add_argument('--pgd-radius', type=float, default=0.0)
parser.add_argument('--pgd-steps', type=int, default=0)
parser.add_argument('--pgd-norm', type=str, default='', choices=['linf', 'l2', ''])
parser.add_argument('--blur-parameter', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--imagenet-dim', type=int, default=224)
parser.add_argument('--name', type=str)
parser.add_argument('--mix', type=float, default=1.0) # percent of poisoned data, default 100% data is poisoned.
args = parser.parse_args()

print(args.arch, args.dataset, args.train_type, args.pgd_radius, args.pgd_steps, args.pgd_norm, args.blur_parameter, args.seed, args.name)

start = time()
train_transform = [transforms.ToTensor()]
test_transform = [transforms.ToTensor()]
train_transform = transforms.Compose(train_transform)
test_transform = transforms.Compose(test_transform)
epochs = 100

if args.dataset == 'cifar10':
    clean_train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)
    clean_test_dataset = datasets.CIFAR10(root='../datasets', train=False, download=True, transform=test_transform)
    num_cls = 10
    size = 32
    batch_size = 512

elif args.dataset == 'cifar100':
    clean_train_dataset = datasets.CIFAR100(root='../datasets', train=True, download=True, transform=train_transform)
    clean_test_dataset = datasets.CIFAR100(root='../datasets', train=False, download=True, transform=test_transform)
    num_cls = 100
    size = 32
    batch_size = 512


clean_train_loader = DataLoader(dataset=clean_train_dataset, batch_size=batch_size,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=4)
                                
clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=512,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=4)


# unlearnable parameters
grayscale = False # grayscale
blur_parameter = args.blur_parameter
center_parameter = 1.0
kernel_size = 3
seed = args.seed
same = False
mix = args.mix

# test parameters
test_grayscale = False


# Below function is the main CUDA algorithm
def get_filter_unlearnable(blur_parameter, center_parameter, grayscale, kernel_size, seed, same):

    np.random.seed(seed)
    cnns = []
    with torch.no_grad():
        for i in range(num_cls): 
            cnns.append(torch.nn.Conv2d(3, 3, kernel_size, groups=3, padding=1).cuda())
            if blur_parameter is None:
                blur_parameter = 1

            w = np.random.uniform(low=0, high=blur_parameter, size=(3,1,kernel_size,kernel_size))
            if center_parameter is not None:
                shape = w[0][0].shape
                w[0, 0, np.random.randint(shape[0]), np.random.randint(shape[1])] = center_parameter

            w[1] = w[0]
            w[2] = w[0]
            cnns[i].weight.copy_(torch.tensor(w))
            cnns[i].bias.copy_(cnns[i].bias * 0)

    cnns = np.stack(cnns)

    if same:
        cnns = np.stack([cnns[0]] * len(cnns))

    if args.dataset == 'cifar10':
        unlearnable_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)

    elif args.dataset == 'cifar100':
        unlearnable_dataset = datasets.CIFAR100(root='../datasets', train=True, download=True, transform=train_transform)


    unlearnable_loader = DataLoader(dataset=unlearnable_dataset, batch_size=500,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False, num_workers=4)

    pbar = tqdm(unlearnable_loader, total=len(unlearnable_loader))
    images_ = []

    for images, labels in pbar:
        images, labels = images.cuda(), labels.cuda()
        for i in range(len(images)):

            prob = np.random.random()
            if prob < mix: # mix*100% of data is poisoned
                id = labels[i].item()
                img = cnns[id](images[i:i+1]).detach().cpu() # convolve class-wise

                # # black and white
                if grayscale:
                    img_bw = img[0].mean(0)
                    img[0][0] = img_bw
                    img[0][1] = img_bw
                    img[0][2] = img_bw    

                images_.append(img/img.max())
            else:
                images_.append(images[i:i+1].detach().cpu())

    # making unlearnable data
    unlearnable_dataset.data = unlearnable_dataset.data.astype(np.float32)
    for i in range(len(unlearnable_dataset)):
        unlearnable_dataset.data[i] = images_[i][0].numpy().transpose((1,2,0))*255
        unlearnable_dataset.data[i] = np.clip(unlearnable_dataset.data[i], a_min=0, a_max=255)
    unlearnable_dataset.data = unlearnable_dataset.data.astype(np.uint8)

    return unlearnable_dataset, cnns

def imshow(img):
    fig = plt.figure(figsize=(9, 3), dpi=250, facecolor='w', edgecolor='k')
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_{}.png'.format(blur_parameter))
    
def get_pairs_of_imgs(idx):
    clean_img = clean_train_dataset.data[idx]
    unlearnable_img = unlearnable_dataset.data[idx]
    clean_img = torchvision.transforms.functional.to_tensor(clean_img)
    unlearnable_img = torchvision.transforms.functional.to_tensor(unlearnable_img)
    noise = unlearnable_img - clean_img
    noise = noise - noise.min()
    noise = noise/noise.max()
    return [clean_img, noise, unlearnable_img]

# def get_altered_testset(cnns, grayscale):

#     pbar = tqdm(clean_test_loader, total=len(clean_test_loader))
#     images_ = []

#     for images, labels in pbar:
#         images, labels = images.cuda(), labels.cuda()
#         for i in range(len(images)):
#             id = labels[i].item()
#             if cnns is None:
#                 img = images[i:i+1].detach().cpu()
#                 images_.append(img)
#                 continue
#             else:
#                 img = cnns[id%10](images[i:i+1]).detach().cpu()
#             if grayscale:
#                 img_bw = img[0].mean(0)
#                 img[0][0] = img_bw
#                 img[0][1] = img_bw
#                 img[0][2] = img_bw   

#             # normalize
#             img = img/img.max()
#             images_.append(img)

#     clean_test_dataset.data = clean_test_dataset.data.astype(np.float32)
#     for i in range(len(clean_test_dataset)):
#         clean_test_dataset.data[i] = images_[i][0].numpy().transpose((1,2,0))*255
#         clean_test_dataset.data[i] = np.clip(clean_test_dataset.data[i], a_min=0, a_max=255)
#     clean_test_dataset.data = clean_test_dataset.data.astype(np.uint8)

#     return clean_test_dataset

unlearnable_dataset, cnns = get_filter_unlearnable(blur_parameter, center_parameter, grayscale, kernel_size, seed, same)
print('Time taken is', time()-start)
# clean_test_dataset = get_altered_testset(None, test_grayscale)

# get unlearnable dataset images
rows = 10
selected_idx = []
for i in range(10):
    idx = (np.stack(clean_train_dataset.targets) == i)
    idx = np.arange(len(clean_train_dataset))[idx]
    np.random.shuffle(idx)
    selected_idx.append(idx[0])
    # print(i, cnns[i].weight.data[0])

# selected_idx = [random.randint(0, len(clean_train_dataset)) for _ in range(rows)]
img_grid = []
for idx in selected_idx:
    img_grid += get_pairs_of_imgs(idx) 

img_grid = img_grid[0::3] + img_grid[1::3] + img_grid[2::3]
imshow(torchvision.utils.make_grid(torch.stack(img_grid), nrow=10, pad_value=255))
# exit()

# get ready for training

train_transform = transforms.Compose([transforms.RandomCrop(size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

if args.dataset == 'cifar10':
    clean_train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)

elif args.dataset == 'cifar100':
    clean_train_dataset = datasets.CIFAR100(root='../datasets', train=True, download=True, transform=train_transform)

unlearnable_dataset.transforms = clean_train_dataset.transforms
unlearnable_dataset.transform = clean_train_dataset.transform

unlearnable_loader = DataLoader(dataset=unlearnable_dataset, batch_size=batch_size,
                                shuffle=True, pin_memory=True,
                                drop_last=False, num_workers=4)


clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=batch_size,
                                shuffle=True, pin_memory=True,
                                drop_last=False, num_workers=4)


# training
arch = args.arch
torch.manual_seed(seed)

if arch == 'resnet18':
    from resnet import resnet18 as net
    model = net(3, num_cls)

elif arch == 'resnet50':
    from resnet import resnet50 as net
    model = net(3, num_cls)

elif arch == 'wrn-34-10':
    from resnet import wrn34_10 as net
    model = net(3, num_cls)

elif arch == 'vgg16-bn':
    from vgg import vgg16_bn as net
    model = net(3, num_cls)

elif arch == 'densenet-121':
    from densenet import densenet121 as net
    model = net(num_classes=num_cls)

model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(epochs*0.4), gamma=0.1)

train_acc = []
test_acc = []

try:

    if args.train_type == 'adv':

        import torchattacks

        if args.pgd_norm == 'linf':
            attacker = torchattacks.PGD
        elif args.pgd_norm == 'l2':
            attacker = torchattacks.PGDL2

        eps = args.pgd_radius
        steps = args.pgd_steps
        atk = attacker(model, eps=eps, alpha=eps/steps * 1.5, steps=steps)


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

            if args.train_type == 'adv':
                images = atk(images, labels)

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            train_acc.append(acc)
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

    with open(args.name, 'wb') as f:
        pkl.dump([train_acc, test_acc], f)

with open(args.name, 'wb') as f:
    pkl.dump([train_acc, test_acc], f)