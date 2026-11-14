"""Group 3 submission: logit-averaged ensemble of member_*.pt models in this folder.

No pretrained weights are downloaded here; every member's weights (fine-tuned from standard ImageNet
ResNet-18 weights or trained from scratch, see the training notebook) are stored in its member file.
Preprocessing (resize, normalisation, optional logit adjustment) lives inside each member's buffers.
"""
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 512
_CACHE = None


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + (x if self.downsample is None else self.downsample(x)))


class ResNet18Body(nn.Module):
    out_dim = 512

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        for i, (a, b, s) in enumerate([(64, 64, 1), (64, 128, 2), (128, 256, 2), (256, 512, 2)], 1):
            setattr(self, f'layer{i}', nn.Sequential(Block(a, b, s), Block(b, b, 1)))

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x)))).mean((2, 3))


class SmallResNet(nn.Module):
    out_dim = 128

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(Block(32, 32, 1), Block(32, 64, 2), Block(64, 128, 2))

    def forward(self, x):
        return self.blocks(self.stem(x)).mean((2, 3))


class Net(nn.Module):
    def __init__(self, arch, num_classes, size, dropout, mean=None, std=None):
        super().__init__()
        self.size = size
        self.body = ResNet18Body() if arch == 'r18' else SmallResNet()
        ch = 3 if arch == 'r18' else 1
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.body.out_dim, num_classes)
        self.register_buffer('mean', torch.tensor(mean if mean is not None else [0.0] * ch, dtype=torch.float32).view(1, ch, 1, 1))
        self.register_buffer('std', torch.tensor(std if std is not None else [1.0] * ch, dtype=torch.float32).view(1, ch, 1, 1))
        self.register_buffer('logit_bias', torch.zeros(num_classes))

    def forward(self, x):
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
        x = (x.expand(-1, self.mean.shape[1], -1, -1) - self.mean) / self.std
        return self.fc(self.drop(self.body(x))) + self.logit_bias


def _device():
    d = os.environ.get('PREDICT_DEVICE')
    if d:
        return torch.device(d)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load():
    global _CACHE
    if _CACHE is None:
        dev = _device()
        paths = sorted(glob.glob(os.path.join(HERE, 'member_*.pt')))
        if not paths:
            raise FileNotFoundError(f'no member_*.pt files in {HERE}')
        nets = []
        for p in paths:
            ck = torch.load(p, map_location='cpu', weights_only=True)
            net = Net(ck['arch'], ck['num_classes'], ck['size'], ck['dropout'])
            net.load_state_dict(ck['state_dict'], strict=True)
            nets.append(net.to(dev).eval())
        _CACHE = (dev, nets)
    return _CACHE


@torch.no_grad()
def predict(images: np.ndarray) -> np.ndarray:
    x = np.asarray(images)
    if x.ndim != 3 or tuple(x.shape[1:]) != (28, 28):
        raise ValueError(f'expected images of shape (N, 28, 28), got {x.shape}')
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    dev, nets = _load()
    out = []
    for i in range(0, x.shape[0], BATCH_SIZE):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + BATCH_SIZE], dtype=np.uint8)).to(dev).float().div(255.0).unsqueeze(1)
        logits = sum(n(xb).float() for n in nets) / len(nets)
        out.append(logits.argmax(dim=1).cpu())
    return torch.cat(out).numpy().astype(np.int64)
