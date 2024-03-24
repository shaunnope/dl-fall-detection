import math

import torch
import torch.nn as nn

from model.modules.conv import ConvLayer, ConvBlock
from utils import dist2bbox, make_anchors


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
      return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
      # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
    

class YOLOHead(nn.Module):
  """YOLOv8 Detect head for detection models."""

  dynamic = False  # force grid reconstruction
  shape = None
  anchors = torch.empty(0)  # init
  strides = torch.empty(0)  # init

  
  def __init__(self, nc=80, ch=()):
    """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
    super().__init__()
    self.nc = nc  # number of classes
    self.nl = len(ch)  # number of detection layers
    self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
    self.no = nc + self.reg_max * 4  # number of outputs per anchor
    self.stride = torch.zeros(self.nl)  # strides computed during build

    c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
    self.cv2 = nn.ModuleList(
        nn.Sequential(ConvLayer(x, c2, 3), ConvLayer(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
    )
    self.cv3 = nn.ModuleList(nn.Sequential(ConvLayer(x, c3, 3), ConvLayer(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
    self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


  def forward(self, x: torch.Tensor):
    """Concatenates and returns predicted bounding boxes and class probabilities."""
    for i in range(self.nl):
      x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:  # Training path
      return x

    # Inference path
    shape = x[0].shape  # BCHW
    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    if self.dynamic or self.shape != shape:
      self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
      self.shape = shape

    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y

  def bias_init(self):
    """Initialize Detect() biases, WARNING: requires stride availability."""
    m = self  # self.model[-1]  # Detect() module
    for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
      a[-1].bias.data[:] = 1.0  # box
      b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

  def decode_bboxes(self, bboxes, anchors):
    """Decode bounding boxes."""
    return dist2bbox(bboxes, anchors, xywh=True, dim=1)



class YOLONet(nn.Module):
  """Backbone of the YOLO model."""

  def __init__(self, s=10, b=2, c=20, ch=3, reg_max=16):
    super(YOLONet, self).__init__()

    self.s = s
    self.nc = c
    
    self.nl = ch
    self.reg_max = reg_max
    self.no = c + self.reg_max * 4

    self.args = {
      "box": 7.5, # (float) box loss gain
      "cls": 0.5, # (float) cls loss gain (scale with pixels)
      "dfl": 1.5, # (float) dfl loss gain
    }

    self.conv_layers = nn.Sequential(
      ConvLayer(3, 64, 7, 2),
      nn.MaxPool2d(2, 2),
      ConvLayer(64, 192, 3, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(192, 128, 256, 3, 1),
      ConvBlock(256, 256, 512, 3, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(512, 256, 512, 3, 1),
      ConvBlock(512, 256, 512, 3, 1),
      ConvBlock(512, 256, 512, 3, 1),
      ConvBlock(512, 256, 512, 3, 1),
      ConvBlock(512, 512, 1024, 3, 1),
      nn.MaxPool2d(2, 2),
      ConvBlock(1024, 512, 1024, 3, 1),
      ConvBlock(1024, 512, 1024, 3, 1),
    ) 

    self.conv_layers2 = nn.Sequential(
      ConvLayer(1024, 1024, 3, 1),
      ConvLayer(1024, 1024, 3, 2),
      ConvLayer(1024, 1024, 3, 1),
      ConvLayer(1024, 1024, 3, 1),
    )

    self.head = nn.Sequential(
      ConvLayer(1024, 1024, 3, 1),
      nn.Conv2d(1024, self.no, 1, 1)
    )

    # self.fc_layers = nn.Sequential(
    #   nn.Linear(1024 * s * s, 4096),
    #   nn.ReLU(),
    #   nn.Linear(4096, s * s * reg_max * (b * 5 + c)),
    #   nn.ReLU()
    # )

    self.stride = 16

  def forward(self, x):
    x = self.conv_layers(x)
    x = self.conv_layers2(x)
    # x = x.view(x.size(0), -1)
    # x = self.fc_layers(x)
    x = self.head(x)
    return x
  
class YOLOPretrainer(nn.Module):
  def __init__(self, model: YOLONet, out_features: int):
    super(YOLOPretrainer, self).__init__()
    self.model = model
    self.conv_layers = model.conv_layers

    self.avg_pool = nn.AvgPool2d(2, 2)
    self.fc = nn.Linear(1024 * model.s * model.s, out_features)

  def forward(self, x):
    out = self.conv_layers(x)
    out = self.avg_pool(out)
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out