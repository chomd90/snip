import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init
from torch.autograd import Variable

from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms

from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar

import datetime
import time

from snip import SNIP
from train_utils import AverageMeter, accuracy, accuracy_list, init_logfile, log
from utils import model_inference, snip_forward_conv2d, snip_forward_linear, train, test, mask_train, apply_prune_mask

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

LOG_INTERVAL = 20
INIT_LR = 0.1
WEIGHT_DECAY_RATE = 0.0005
EPOCHS = 160
REPEAT_WITH_DIFFERENT_SEED = 3

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def apply_prune_mask(net, keep_masks):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask))


class LeNet_300_100(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 784)))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc3 = nn.Linear(16 * 5 * 5, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.fc3(x.view(-1, 16 * 5 * 5)))
        x = F.relu(self.fc4(x))
        x = F.log_softmax(self.fc5(x))

        return x


class LeNet_5_Caffe(nn.Module):
    """
    This is based on Caffe's implementation of Lenet-5 and is slightly different
    from the vanilla LeNet-5. Note that the first layer does NOT have padding
    and therefore intermediate shapes do not match the official LeNet-5.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, padding=0)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc3 = nn.Linear(50 * 4 * 4, 500)
        self.fc4 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.fc3(x.view(-1, 50 * 4 * 4)))
        x = F.log_softmax(self.fc4(x))

        return x


VGG_CONFIGS = {
    # M for MaxPool, Number for channels
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
}


class VGG_SNIP(nn.Module):
    """
    This is a base class to generate three VGG variants used in SNIP paper:
        1. VGG-C (16 layers)
        2. VGG-D (16 layers)
        3. VGG-like

    Some of the differences:
        * Reduced size of FC layers to 512
        * Adjusted flattening to match CIFAR-10 shapes
        * Replaced dropout layers with BatchNorm
    """

    def __init__(self, config, num_classes=10):
        super().__init__()

        self.features = self.make_layers(VGG_CONFIGS[config], batch_norm=True)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  # 512 * 7 * 7 in the original VGG
            nn.ReLU(True),
            nn.BatchNorm1d(512),  # instead of dropout
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),  # instead of dropout
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def make_layers(config, batch_norm=False):  # TODO: BN yes or no?
        layers = []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def get_mnist_dataloaders(train_batch_size, val_batch_size):

    data_transform = Compose([transforms.ToTensor()])

    # Normalise? transforms.Normalize((0.1307,), (0.3081,))

    train_dataset = MNIST("_dataset", True, data_transform, download=True)
    test_dataset = MNIST("_dataset", False, data_transform, download=False)

    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        val_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    return train_loader, test_loader


def get_cifar10_dataloaders(train_batch_size, test_batch_size):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10('_dataset', True, train_transform, download=True)
    test_dataset = CIFAR10('_dataset', False, test_transform, download=False)

    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    return train_loader, test_loader


def mnist_experiment():

    BATCH_SIZE = 100
    LR_DECAY_INTERVAL = 25000

    # net = LeNet_300_100()
    # net = LeNet_5()
    net = LeNet_5_Caffe().to(device)

    optimiser = optim.SGD(
        net.parameters(),
        lr=INIT_LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(optimiser, 30000, gamma=0.1)

    train_loader, val_loader = get_mnist_dataloaders(BATCH_SIZE, BATCH_SIZE)

    return net, optimiser, lr_scheduler, train_loader, val_loader


def cifar10_experiment():

    BATCH_SIZE = 128
    LR_DECAY_INTERVAL = 30000

    net = VGG_SNIP('D').to(device)
#     net = ResNet(BasicBlock, [3, 3, 3]).to(device)
    optimiser = optim.SGD(
        net.parameters(),
        lr=INIT_LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimiser, LR_DECAY_INTERVAL, gamma=0.1)

    train_loader, val_loader = get_cifar10_dataloaders(BATCH_SIZE,
                                                       BATCH_SIZE)  # TODO

    return net, optimiser, lr_scheduler, train_loader, val_loader


def main():

#     writer = SummaryWriter()
    test_acc_list = []
    logfilename = os.path.join('.', 'log3.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")


#     net = ResNet(BasicBlock, [3, 3, 3]).to(device)

    net = VGG_SNIP('D').to(device)
#     criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = SGD(net.parameters(), lr=0.1, 
                        momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[80, 120], last_epoch=-1)
    train_loader, test_loader = get_cifar10_dataloaders(128, 128)
    
    keep_masks = SNIP(net, 0.05, train_loader, device)  # TODO: shuffle?
    apply_prune_mask(net, keep_masks)
    
    for epoch in range(160):
        before = time.time()
        train_loss, train_acc = train(train_loader, net, criterion, 
                                      optimizer, epoch, device, 
                                      100, display=True)
        test_loss, test_acc = test(test_loader, net, criterion, device, 
                                   100, display=True)

        scheduler.step(epoch)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))


        print("test_acc: ", test_acc)
    test_acc_list.append(test_acc)
    log(logfilename, "This is the test accuracy list for args.round.")
    log(logfilename, str(test_acc_list))
    
    
    
    
    
    
    

#     trainer = create_supervised_trainer(net, optimiser, F.nll_loss, device)
#     evaluator = create_supervised_evaluator(net, {
#         'accuracy': Accuracy(),
#         'nll': Loss(F.nll_loss)
#     }, device)

#     pbar = ProgressBar()
#     pbar.attach(trainer)

#     @trainer.on(Events.ITERATION_COMPLETED)
#     def log_training_loss(engine):
#         lr_scheduler.step()
#         iter_in_epoch = (engine.state.iteration - 1) % len(train_loader) + 1
#         if engine.state.iteration % LOG_INTERVAL == 0:
#             # pbar.log_message("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
#             #       "".format(engine.state.epoch, iter_in_epoch, len(train_loader), engine.state.output))
#             writer.add_scalar("training/loss", engine.state.output,
#                               engine.state.iteration)

#     @trainer.on(Events.EPOCH_COMPLETED)
#     def log_epoch(engine):
#         evaluator.run(val_loader)

#         metrics = evaluator.state.metrics
#         avg_accuracy = metrics['accuracy']
#         avg_nll = metrics['nll']

#         # pbar.log_message("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
#         #       .format(engine.state.epoch, avg_accuracy, avg_nll))

#         writer.add_scalar("validation/loss", avg_nll, engine.state.iteration)
#         writer.add_scalar("validation/accuracy", avg_accuracy,
#                           engine.state.iteration)

#     trainer.run(train_loader, EPOCHS)

#     # Let's look at the final weights
#     # for name, param in net.named_parameters():
#     #     if name.endswith('weight'):
#     #         writer.add_histogram(name, param)

#     writer.close()


if __name__ == '__main__':
    main()
