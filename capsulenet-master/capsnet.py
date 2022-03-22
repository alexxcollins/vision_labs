import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable

from capsule_layer import CapsuleLayer
from convrelu import Conv2dRelu


class CapsNet(nn.Module):
    def __init__(self, n_conv_channel, n_primary_caps, primary_cap_size,
                 output_unit_size, n_routing_iter):

        super(CapsNet, self).__init__()

        self.conv1 = Conv2dRelu(1, n_conv_channel, 9)

        self.primary = CapsuleLayer(0,
                                    n_conv_channel,
                                    n_primary_caps,
                                    primary_cap_size,
                                    False,
                                    n_routing_iter)

        self.final_caps = CapsuleLayer(n_primary_caps,
                                       primary_cap_size,
                                       10,  # 10 catagories in MNIST
                                       output_unit_size,
                                       True,
                                       n_routing_iter)

    def forward(self, X):
        X = self.conv1(X)
        X = self.primary(X)
        X = self.final_caps(X)
        return X

    def loss(self, pred, target, size_average=True):
        """
        Margin loss for digit existence described in eq.4
        params:
        - pred: output from digit caps (16x10 for mnist)
        - target: one-hot encoding of class
        """
        batch_size = pred.size(0)

        # ||v_k||
        v_k = torch.sqrt((pred**2).sum(dim=2, keepdim=True))

        zero = torch.zeros(1)
        zero = zero.cuda() if pred.is_cuda else zero

        # constants defined in paper
        m_plus = 0.9
        m_minus = 0.1
        loss_lambda = 0.5

        # Tk max(0, m+ − ||v_k||)^2 (left side of eq 4)
        left = torch.max(zero, m_plus - v_k).view(batch_size, -1)

        # λ (1 − T_k) max(0, ||vk|| − m_minus)^2 (right side of eq 4)
        right = torch.max(zero, v_k - m_minus).view(batch_size, -1)

        t_k = target

        # L_k is margin loss for each digit of class k
        l_k = t_k * left + loss_lambda * (1.0 - t_k) * right
        l_k = l_k.sum(dim=1)

        if size_average:
            l_k = l_k.mean()

        return l_k

    def capsule_prediction(self, final_caps):
        '''
        Converts output of last caps layer to softmax probability of each digit
        params:
        - final_caps: output from final layer, in mnist the shape is (16, 10)
        '''
        # ||D|| where D is the digit caps
        return F.softmax(torch.sum(final_caps**2,
                                   dim=2, keepdim=True)).view(-1, 10)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.l1 = nn.Linear(160, 512)
        self.l2 = nn.Linear(512, 1024)
        self.l3 = nn.Linear(1024, 784)

    def forward(self, X):
        X = X.view(-1, 160)
        X = F.relu(self.l1(X))
        X = F.relu(self.l2(X))
        X = F.sigmoid(self.l3(X))
        X = X.view(-1, 1, 28, 28)
        return X
