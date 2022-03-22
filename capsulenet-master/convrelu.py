import torch.nn as nn
import torch.nn.functional as F


class Conv2dRelu(nn.Module):
    '''Standard 2D Conv with relu activation'''
    def __init__(self, n_channels_in, n_channels_out, kernel):
        super(Conv2dRelu, self).__init__()

        self.conv = nn.Conv2d(n_channels_in,
                              n_channels_out,
                              kernel,
                              stride=1)

    def forward(self, X):
        return F.relu(self.conv(X))
