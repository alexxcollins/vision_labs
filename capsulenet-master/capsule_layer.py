import torch
import torch.nn as nn
#from torch.autograd import Variable
import torch.nn.functional as F


class CapsuleLayer(nn.Module):

    def __init__(self, n_units_prev, in_channels, n_unit, unit_size,
                 use_routing, n_routing_iters):
        '''
        params:
        n_units_prev (int): number of capsule units in previous layer
        in_channels (int): number of channels on the tensor fed into capsule
        n_units (int): number of capsule units
        unit_size (int): size of each capsule
        use_routing (bool): if layer is preceeded by layer of caps, use routing
        n_routing_iters (int): if routing, how many iters to apply routing alg
        '''
        super(CapsuleLayer, self).__init__()

        self.n_units_prev = n_units_prev
        self.in_channels = in_channels
        self.n_unit = n_unit
        self.unit_size = unit_size
        self.use_routing = use_routing
        self.n_routing_iters = n_routing_iters

        if self.use_routing:
            self.W = nn.Parameter(torch.randn(1, in_channels, n_unit,
                                              unit_size, n_units_prev))

        else:
            self.unit_convs = []
            for i in range(self.n_unit):
                # >in_channels, 32 out, kernel of 9, stride of 2
                c = nn.Conv2d(in_channels, 32, 9, 2)

                self.add_module('conv_unit_{}'.format(i), c)
                self.unit_convs.append(c)

    def squash(self, sj):
        """
        Squashing function described in Equation 1
        """
        sj_norm_sqrd = torch.sum(sj**2, dim=2, keepdim=True)
        sj_norm = torch.sqrt(sj_norm_sqrd)
        v_j = (sj_norm_sqrd / (1.0 + sj_norm_sqrd)) * (sj / sj_norm)
        return v_j

    def forward(self, X):
        if self.use_routing:
            return self.routing(X)

        else:
            # If this layer is not preceeded by a CapsuleLayer
            # apply each capsule's conv to the input,
            # reshape the output from each conv into a vector capsule
            units = [self.unit_convs[i](X) for i in range(self.n_unit)]
            units = torch.stack(units, dim=1)
            units = units.view(X.size(0), self.n_unit, -1)
            # Apply non-linearity
            return self.squash(units)

    def routing(self, X):
        # X is the output from the previous layer l.
        # This layer is considered l+1 by the paper as routing is defined
        # between layer l and l+1
        batch_size = X.size(0)

        X = X.transpose(1, 2)
        X = torch.stack([X] * self.n_unit, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size)

        # apply weights
        u_hat = torch.matmul(W, X)

        # Routing described in Precedure 1

        # Precedure 1 line 2: b_ij <- 0
        b_ij = torch.zeros(1, self.in_channels, self.n_unit, 1)
        b_ij = b_ij.cuda() if X.is_cuda else b_ij

        # Precedure 1 line 3
        for i in range(self.n_routing_iters):
            # Precedure 1 line 4 c_i <- softmax(b_i)
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size).unsqueeze(4)

            # Precedure 1 line 5 s_j <- sum_i( c_ij * u_j|i )
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # Precedure 1 line 6 squash using eq.1
            v_j = self.squash(s_j)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # Precedure 1 line 7 u_j|i * v_j
            uv_j = torch.matmul(u_hat.transpose(3, 4), v_j1)
            uv_j = uv_j.squeeze(4).mean(dim=0, keepdim=True)

            # Precedure 1 line 7 b_ij <- b_ij + u_j|i * v_j
            b_ij = uv_j + b_ij

        return v_j.squeeze(1)
