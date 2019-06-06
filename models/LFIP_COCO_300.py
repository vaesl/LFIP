import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(ConvBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ExtraBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(ExtraBlock, self).__init__()
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.single_branch = nn.Sequential(
                ConvBlock(in_planes, inter_planes, kernel_size=1),
                ConvBlock(inter_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=1, relu=False)
                )

    def forward(self, x):
        out = self.single_branch(x)
        return out


class LFIP(nn.Module):

    def __init__(self, in_planes):
        super(LFIP, self).__init__()
        self.iter_ds = Iter_Downsample()
        self.lcb1 = nn.Sequential(
            ConvBlock(in_planes, 128, kernel_size=(3, 3), padding=1), ConvBlock(128, 128, kernel_size=1, stride=1),
            ConvBlock(128, 128, kernel_size=(3, 3), padding=1), ConvBlock(128, 512, kernel_size=1, relu=False))
        self.lcb2 = nn.Sequential(
            ConvBlock(in_planes, 256, kernel_size=(3, 3), padding=1), ConvBlock(256, 256, kernel_size=1),
            ConvBlock(256, 256, kernel_size=(3, 3), padding=1), ConvBlock(256, 256, kernel_size=1, stride=1),
            ConvBlock(256, 256, kernel_size=(3, 3), padding=1), ConvBlock(256, 1024, kernel_size=1, relu=False))
        self.lcb3 = nn.Sequential(
            ConvBlock(in_planes, 128, kernel_size=(3, 3), padding=1), ConvBlock(128, 128, kernel_size=1),
            ConvBlock(128, 128, kernel_size=(3, 3), padding=1), ConvBlock(128, 128, kernel_size=1),
            ConvBlock(128, 128, kernel_size=(3, 3), padding=1), ConvBlock(128, 512, kernel_size=1, relu=False))
        self.lcb4 = nn.Sequential(
            ConvBlock(in_planes, 64, kernel_size=(3, 3), padding=1), ConvBlock(64, 64, kernel_size=1),
            ConvBlock(64, 64, kernel_size=(3, 3),  padding=1), ConvBlock(64, 64, kernel_size=1),
            ConvBlock(64, 64, kernel_size=(3, 3), padding=1), ConvBlock(64, 256, kernel_size=1, relu=False))

    def forward(self, x):
        img1, img2, img3, img4 = self.iter_ds(x)
        s1 = self.lcb1(img1)
        s2 = self.lcb2(img2)
        s3 = self.lcb3(img3)
        s4 = self.lcb4(img4)
        return s1, s2, s3, s4


class Iter_Downsample(nn.Module):

    def __init__(self,):
        super(Iter_Downsample, self).__init__()
        self.init_ds = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.ds1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.ds2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.ds3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.ds4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.init_ds(x)
        x1 = self.ds1(x)
        x2 = self.ds2(x1)
        x3 = self.ds3(x2)
        x4 = self.ds4(x3)
        return x1, x2, x3, x4


class FAM(nn.Module):

    def __init__(self, plane1, plane2, bn=True, ds=True):
        super(FAM, self).__init__()
        self.bn = nn.BatchNorm2d(plane2, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.dsc = ConvBlock(plane1, plane2, kernel_size=(3, 3), stride=2, padding=1, relu=False) if ds else None
        self.merge = ConvBlock(plane2, plane2, kernel_size=(3, 3), stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, o, s, p):

        o_bn = self.bn(o) if self.bn is not None else o
        s_bn = s

        x = o_bn * s_bn + self.dsc(p) if self.dsc is not None else o_bn * s_bn
        out = self.merge(self.relu(x))

        return out


class LFIPNet(nn.Module):
    """LFIP Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(LFIPNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # vgg network
        self.base = nn.ModuleList(base)

        self.lfip = LFIP(in_planes=3)

        self.fam1 = FAM(plane1=512, plane2=512, bn=True, ds=False)
        self.fam2 = FAM(plane1=512, plane2=1024, bn=True, ds=True)
        self.fam3 = FAM(plane1=1024, plane2=512, bn=False, ds=True)
        self.fam4 = FAM(plane1=512, plane2=256, bn=False, ds=True)

        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax()

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # generate image pyramid
        s1, s2, s3, s4 = self.lfip(x)

        # apply vgg up to conv4_3
        for k in range(22):
            x = self.base[k](x)
        f1 = self.fam1(x, s1, None)
        sources.append(f1)

        # apply vgg up to fc7
        for k in range(22, 34):
            x = self.base[k](x)
        f2 = self.fam2(x, s2, f1)
        sources.append(f2)

        x = self.base[34](x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k == 0:
                f3 = self.fam3(x, s3, f2)
                sources.append(f3)
            elif k == 2:
                f4 = self.fam4(x, s4, f3)
                sources.append(f4)
            elif k == 5 or k == 7:
                sources.append(x)
            else:
                pass

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=False), conv7, nn.ReLU(inplace=False)]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


def add_extras(size, cfg, i):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256 and size == 512:
                    layers += [ExtraBlock(in_channels, cfg[k+1], stride=2), nn.ReLU(inplace=False)]
                else:
                    layers += [ExtraBlock(in_channels, cfg[k+1], stride=2), nn.ReLU(inplace=False)]
        in_channels = v
    if size == 512:
        layers += [ConvBlock(256, 128, kernel_size=1, stride=1)]
        layers += [ConvBlock(128, 256, kernel_size=4, stride=1, padding=1)]
    elif size == 300:
        layers += [ConvBlock(256, 128, kernel_size=1,stride=1)]
        layers += [ConvBlock(128, 256, kernel_size=3,stride=1)]
        layers += [ConvBlock(256, 128, kernel_size=1,stride=1)]
        layers += [ConvBlock(128, 256, kernel_size=3,stride=1)]
    else:
        print("Error: Sorry only LFIPNet300 and LFIPNet512 are supported!")
        return
    return layers

extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256],
}


def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [1, -2]
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512, cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers +=[nn.Conv2d(512, cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    i = 2
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only LFIPNet300 and LFIPNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if (k < indicator+1 and k % 2 == 0) or (k > indicator+1 and k % 2 != 0):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i] * num_classes, kernel_size=3, padding=1)]
            i += 1
    return vgg, extra_layers, (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only LFIPNet300 and LFIPNet512 are supported!")
        return

    return LFIPNet(phase, size, *multibox(size, vgg(base[str(size)], 3),
                                add_extras(size, extras[str(size)], 1024),
                                mbox[str(size)], num_classes), num_classes)
