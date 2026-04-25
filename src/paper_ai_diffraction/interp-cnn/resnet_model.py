import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

# ---------------------- Output Wrapper ----------------------
class ModelOutput:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss
        self.predictions = torch.flatten(torch.argmax(logits, dim=-1))

    def __str__(self):
        return str({"loss": self.loss, "predictions": self.predictions, "logits": self.logits})

    def accuracy(self, labels):
        assert labels.shape == self.predictions.shape, "Predictions and labels do not have the same shape"
        accuracy = (torch.sum((self.predictions == labels)) / len(self.predictions)).item()
        return round(accuracy, 4) * 100

    def top_k_preds(self, k):
        return torch.topk(self.logits, dim=-1, k=k).indices

    def top_k_acc(self, labels, k):
        labels = torch.unsqueeze(labels, dim=-1)
        labels = torch.cat(tuple(labels for _ in range(k)), -1)
        labels = torch.unsqueeze(labels, dim=1)
        preds = self.top_k_preds(k)
        acc = (torch.sum(preds == labels) / len(labels)).item()
        return round(acc, 4) * 100

# ---------------------- CNN Layers ----------------------
class CNN1dlayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_p=0.1, bias=False, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias, padding=padding)
        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(out_channels, elementwise_affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = x.transpose(-2, -1)
        x = self.layer_norm(x)
        x = x.transpose(-2, -1)
        return self.act(x)

class CNN1dlayerNoAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_p=0.1, bias=False, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias, padding=padding)
        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(out_channels, elementwise_affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = x.transpose(-2, -1)
        x = self.layer_norm(x)
        x = x.transpose(-2, -1)
        return x

# ---------------------- Blocks ----------------------
class BasicBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_p=0.05, downsample=False):
        super().__init__()
        self.conv1 = CNN1dlayer(in_channels, out_channels, kernel_size, stride, dropout_p=dropout_p, padding=kernel_size // 2)
        self.conv2 = CNN1dlayerNoAct(out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout_p=dropout_p, padding=kernel_size // 2)
        self.downsample = CNN1dlayerNoAct(in_channels, out_channels, kernel_size=1, stride=stride, dropout_p=0) if downsample else None
        self.act = nn.GELU()

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + identity)

class Bottleneck1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_p=0.05, downsample=False):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = CNN1dlayer(in_channels, mid_channels, kernel_size=1, stride=1, dropout_p=dropout_p)
        self.conv2 = CNN1dlayer(mid_channels, mid_channels, kernel_size=3, stride=stride, dropout_p=dropout_p, padding=1)
        self.conv3 = CNN1dlayerNoAct(mid_channels, out_channels, kernel_size=1, stride=1, dropout_p=dropout_p)
        self.downsample = CNN1dlayerNoAct(in_channels, out_channels, kernel_size=1, stride=stride, dropout_p=0) if downsample or in_channels != out_channels else None
        self.act = nn.GELU()

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.act(x + identity)

# ---------------------- Config ----------------------
class ResnetConfig:
    def __init__(self, input_dim=1, output_dim=14, res_dims=[64, 128, 256, 512], res_kernel=[3, 3, 3, 3],
                 res_stride=[1, 2, 2, 2], num_blocks=[2, 2, 2, 2], first_kernel_size=7, first_stride=2,
                 first_pool_kernel_size=3, first_pool_stride=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.res_dims = res_dims
        self.res_kernel = res_kernel
        self.res_stride = res_stride
        self.num_blocks = num_blocks
        self.first_kernel_size = first_kernel_size
        self.first_stride = first_stride
        self.first_pool_kernel_size = first_pool_kernel_size
        self.first_pool_stride = first_pool_stride

# ---------------------- ResNet Body ----------------------
class Resnet(nn.Module):
    def __init__(self, config, block_type="basic"):
        super().__init__()
        self.block_class = BasicBlock1d if block_type == "basic" else Bottleneck1dBlock

        self.conv = CNN1dlayer(config.input_dim, config.res_dims[0], kernel_size=config.first_kernel_size,
                               stride=config.first_stride, dropout_p=0.05)
        self.maxpool = nn.MaxPool1d(config.first_pool_kernel_size, stride=config.first_pool_stride,
                                    padding=config.first_pool_kernel_size // 2)

        self.layer1 = self._make_resnet_layer(self.block_class, config.res_dims[0], config.res_dims[0], config.num_blocks[0], config.res_kernel[0], config.res_stride[0])
        self.layer2 = self._make_resnet_layer(self.block_class, config.res_dims[1], config.res_dims[0], config.num_blocks[1], config.res_kernel[1], config.res_stride[1])
        self.layer3 = self._make_resnet_layer(self.block_class, config.res_dims[2], config.res_dims[1], config.num_blocks[2], config.res_kernel[2], config.res_stride[2])
        self.layer4 = self._make_resnet_layer(self.block_class, config.res_dims[3], config.res_dims[2], config.num_blocks[3], config.res_kernel[3], config.res_stride[3])

    def _make_resnet_layer(self, block, out_channels, in_channels, num_blocks, kernel_size=3, stride=1, dropout_p=0.05):
        layers = [block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                        stride=stride, dropout_p=dropout_p, downsample=True)]
        for _ in range(1, num_blocks):
            layers.append(block(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=1, dropout_p=dropout_p, downsample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# ---------------------- Classifier ----------------------
class ResnetClassifier(nn.Module):
    def __init__(self, input_dim, res_dims, res_kernel, res_stride, num_blocks, first_kernel_size,
                 first_stride, first_pool_kernel_size, first_pool_stride, num_classes=None, block_type="basic"):

        config = ResnetConfig(input_dim=input_dim, output_dim=num_classes, res_dims=res_dims, res_kernel=res_kernel,
                              res_stride=res_stride, num_blocks=num_blocks, first_kernel_size=first_kernel_size,
                              first_stride=first_stride, first_pool_kernel_size=first_pool_kernel_size,
                              first_pool_stride=first_pool_stride)

        super().__init__()
        self.resnet = Resnet(config, block_type=block_type)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(config.res_dims[-1], config.output_dim)
        self.adv = nn.Linear(config.res_dims[-1], 1)
        self.num_labels = config.output_dim

    def forward(self, inputs, labels=None, s=True, loss_func=None):
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        x = self.resnet(inputs)
        pooled = self.avgpool(x).squeeze(-1)
        s_logits = self.classifier(pooled)
        u_logits = self.adv(pooled)

        if labels is not None:
            if s:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(s_logits.view(-1, self.num_labels), labels.view(-1))
                return s_logits
            loss = loss_func(u_logits, labels)
            return u_logits
        return s_logits
