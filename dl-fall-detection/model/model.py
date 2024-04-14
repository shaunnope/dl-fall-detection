import math

import torch
import torch.nn as nn
from model.modules.conv import ConvBlock, ConvLayer, ResidualConv
from utils import dist2bbox, make_anchors, bbox_iou

# region: Dive into Deep Learning 
## http://d2l.ai/chapter_computer-vision/anchor.html

def nms(boxes, scores, iou_threshold=0.5):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = bbox_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


# endregion

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )


class DetectHead(nn.Module):
    """Detection head adapted from YOLOv8."""

    dynamic = False  # force grid reconstruction
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initialize the detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = (
            16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        )
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(
            ch[0], min(self.nc, 100)
        )  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                ConvLayer(x, c2, 3),
                ConvLayer(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                ConvLayer(x, c3, 3), ConvLayer(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.args = {
            "box": 7.5,  # (float) box loss gain
            "cls": 0.5,  # (float) cls loss gain (scale with pixels)
            "dfl": 1.5,  # (float) dfl loss gain
        }

    def forward(self, x, encoded=False):
        """Concatenates and returns predicted bounding boxes and class probabilities."""

        x = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i]).log_softmax(1)), 1) for i in range(self.nl)]
        if self.training or encoded:  # Training path
            return x
        
        return self.decode(x)
    
    def decode(self, x):
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = (
            self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        )
        return dbox, cls

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(
                5 / m.nc / (640 / s) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


class YOLOBackbone(nn.Module):
    """Backbone of the YOLO model."""

    def __init__(self, input_ch=3, ch=(64,), hidden_ch=512, dropout=0.2):
        super(YOLOBackbone, self).__init__()

        self.nl = len(ch)
        ch = (hidden_ch, *ch)

        self.conv_layers = nn.Sequential(
            ConvLayer(input_ch, 64, 7, 2),
            ConvLayer(64, 192, 3, 2),
            ResidualConv(192, 192, 3, 1, dropout=dropout),
            ConvBlock(192, 128, 1024, 3, 1),
            # ConvBlock(256, 256, 512, 3, 1),
            # ConvBlock(512, 512, 1024, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvLayer(1024, hidden_ch, 3, 1),
            # ResidualConv(512, 512, 3, 1),
            nn.Dropout(dropout),
        )

        self.detection_layers = [
            ConvLayer(ch[i], ch[i + 1], 3, 2) for i in range(len(ch) - 1)
        ]

    def forward(self, x):
        x = self.conv_layers(x)
        out = [None for _ in range(self.nl)]
        prev = x
        for i, layer in enumerate(self.detection_layers):
            prev = layer(prev)
            out[i] = prev
        return out


class YOLONet(nn.Module):

    def __init__(self, nc=80, ch=(64,), input_ch=3, hidden_ch=512, dropout=0.2):
        super(YOLONet, self).__init__()
        self.backbone = YOLOBackbone(ch=ch, input_ch=input_ch, hidden_ch=hidden_ch, dropout=dropout)
        self.head = DetectHead(nc=nc, ch=ch)

        self.build()

    def forward(self, x, encoded=False):
        x = self.backbone(x)
        out = self.head(x, encoded)
        return out

    def build(self, ch=3):
        s = 256

        self.stride = torch.tensor(
            [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
        )
        self.head.stride = self.stride
        self.head.bias_init()


if __name__ == "__main__":
    head = DetectHead()

    print(head)

    print(head(torch.randn(1, 3, 256, 256)))
