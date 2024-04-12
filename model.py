import torch
import torch.nn as nn

class SleepClassifier(torch.nn.Module):

    def __init__(self):
      super(SleepClassifier, self).__init__()

      self.conv1 = nn.Conv1d(1, 8, kernel_size=1, stride=1)

      self.ConvBlock1 = ConvBlock(8, 16)
      self.ConvBlock2 = ConvBlock(16, 32)
      self.ConvBlock3 = ConvBlock(32, 64)

      self.fc1 = nn.Linear(64*32, 128)

      self.diconv1 = OuterDiConvBlock(128)
      self.diconv2 = OuterDiConvBlock(128)

      self.conv2 = nn.Conv1d(128, 4, kernel_size=1, padding='same')



    def forward(self, x):
      out = self.conv1(x)
      out = self.ConvBlock1(out)
      out = self.ConvBlock2(out)
      out = self.ConvBlock3(out)

      flattened = out.flatten(start_dim=1)
      out = self.fc1(flattened)

      # As per paper, reshape batch dimension to time dimension and embedding
      # nodes to channels
      out = out.view(1, 128, 1200)

      out = self.diconv1(out)
      out = self.diconv2(out)

      out = self.conv2(out)
      return out


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels=None):
    super(ConvBlock, self).__init__()

    # Assumption I needed to make on out_channels between the two Conv1d layers
    if not mid_channels:
      mid_channels=out_channels

    self.l_relu1 = nn.LeakyReLU(0.15)
    self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding='same')

    self.l_relu2 = nn.LeakyReLU(0.15)
    self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding='same')
    self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    # Input needs to be transformed to the same number of channels, assumption
    # was to use a Conv1d to do so, other methods can be stacking the channel as well
    self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)


  def forward(self, x):
    res = self.residual(x)

    out = self.l_relu1(x)
    out = self.conv1(out)

    out = self.l_relu2(out)
    out = self.conv2(out)

    # Paper is unclear whether they take the MaxPool of sum or sum of MaxPool,
    # I chose the latter
    out = self.pool(out)
    out += self.pool(res)

    return out

class OuterDiConvBlock(nn.Module):
  def __init__(self, n_channels):
    super(OuterDiConvBlock, self).__init__()

    # Paper is unclear on out_channel, I assumed it's a power of 2 stepwise
    self.diconv1 = InnerDiConvBlock(n_channels, 2)
    self.diconv2 = InnerDiConvBlock(n_channels, 4)
    self.diconv3 = InnerDiConvBlock(n_channels, 8)
    self.diconv4 = InnerDiConvBlock(n_channels, 16)
    self.diconv5 = InnerDiConvBlock(n_channels, 32)
    self.dropout = nn.Dropout(p=0.2)

  def forward(self, x):
    out = self.diconv1(x)
    out = self.diconv2(out)
    out = self.diconv3(out)
    out = self.diconv4(out)
    out = self.diconv5(out)

    out = self.dropout(out)
    out += x # residual connection

    return out

class InnerDiConvBlock(nn.Module):
  def __init__(self, n_channels, dilation=1):
    super(InnerDiConvBlock, self).__init__()

    self.l_relu = nn.LeakyReLU(0.15)
    self.conv1 = nn.Conv1d(n_channels, n_channels, kernel_size=7,
                           padding='same',dilation=dilation)


  def forward(self, x):
    out = self.l_relu(x)
    out = self.conv1(x)

    return out