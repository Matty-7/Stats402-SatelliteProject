from __future__ import print_function
import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR



class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer.
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.fcn_resnet = torchvision.models.segmentation.fcn_resnet50(True)
        self.fcn_resnet.classifier[4] = nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))
        self.fcn_resnet.aux_classifier[4] = nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))

        self.sigmoid = nn.Sigmoid()
        self.model00 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 24, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        ).to('cuda')

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        # output = self.resnet(x)
        output = self.fcn_resnet(x)
        # output = output.view(output.size()[0], -1)
        # print(type(output))
        # print(output)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)['out']
        output2 = self.forward_once(input2)['out']

        # concatenate both images' features
        # print('beforecat', output1.shape)
        output = torch.cat((output1, output2), 1)
        # print('cat', output.shape)
        output = self.model00(output)


        # pass the concatenation to the linear layers
        print(output.shape)
        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output



import os

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, transform=None, target_transform=None):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.fname_list = None
        self.base_dir = base_dir
        self.target_transform = target_transform
        self.create_fname_list()
        self.img_labels = self.fname_list

    def create_fname_list(self):
        self.fname_list = os.listdir(os.path.join(self.base_dir, '2012'))
        self.fname_list = list(filter(lambda x: x.find("jpg") != -1, self.fname_list))
        self.fname_list = list(map(lambda x: x[5: ], self.fname_list))

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # TODO: refactor
        img_path = os.path.join(self.base_dir, '2012', '2012_' + self.fname_list[idx])
        img_path2 = os.path.join(self.base_dir, '2016', '2016_' + self.fname_list[idx])
        image = plt.imread(img_path)
        image2 = plt.imread(img_path2)
        p2 = np.array(np.zeros((3, 256, 256)))
        p2[0, ...] = image[..., 0]
        p2[1, ...] = image[..., 1]
        p2[2, ...] = image[..., 2]
        p3 = np.array(np.zeros((3, 256, 256)))
        p3[0, ...] = image2[..., 0]
        p3[1, ...] = image2[..., 1]
        p3[2, ...] = image2[..., 2]
        label_path = os.path.join(self.base_dir, 'label', 'label_' + self.fname_list[idx])
        image_label = plt.imread(label_path)
        label = np.divide(image_label[..., 0], 255).astype(int)
        # lab = np.array(np.zeros((1, 256, 256)))
        # lab[0, :, :] = label
        tor = torch.from_numpy(p2)
        tor2 = torch.from_numpy(p3)
        feature = self.normalize_image(tor)
        feature2 = self.normalize_image(tor2)

        return feature, feature2, torch.from_numpy(label).float()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # for the 1st epoch, the average loss is 0.0001 and the accuracy 97-98%
    # using default settings. After completing the 10th epoch, the average
    # loss is 0.0000 and the accuracy 99.5-100% using default settings.
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=25, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # train_dataset = APP_MATCHER('../data', train=True, download=True)
    train_dataset = CustomImageDataset('images/largechange')
    # test_dataset = APP_MATCHER('../data', train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    # test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        print(epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "siamese_network0.pt")


if __name__ == '__main__':
    main()
