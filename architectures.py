import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.cifar_resnet import lenet300
from archs.cifar_resnet import lenet5
from archs.cifar_resnet import fcn, wide_resnet
from archs.cifar_resnet import vgg, resnet20, resnet32, vgg_16
from torch.nn.functional import interpolate

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet32", "cifar_resnet110", "lenet300", "lenet5", "fcn", "wide_resnet", "vgg19",
                 "resnet20", "resnet32", "vgg16"]
def get_architecture(arch: str, dataset: str, device) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).to(device)
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).to(device)
    elif arch == "cifar_resnet32":
        model = resnet_cifar(depth=32, num_classes=10).to(device)
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).to(device)
    elif arch == "lenet300":
        model = lenet300(num_classes=10).to(device)
    elif arch == "lenet5":
        model = lenet5(num_classes=10).to(device)
    elif arch == "fcn":
        model = fcn(num_classes=10).to(device)
    elif arch == "vgg19":
        model = vgg().to(device)
    elif arch == "vgg16":
        model = vgg_16().to(device)
    elif arch == "wide_resnet":
        model = wide_resnet().to(device)
    elif arch == "resnet20":
        model = resnet20().to(device)
    elif arch == "resnet32":
        model = resnet32().to(device)

    return model