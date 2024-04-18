import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn import ConvModule
# from mmcv.cnn import build_norm_layer, build_activation_layer
from torch.autograd import Variable
import numpy as np
import math
from model.dropSke import DropBlock_Ske
from model.dropT import DropBlockT_1d


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2)))
    nn.init.constant(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, num_point=25, block_size=41):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = build_norm_layer(dict(type='BN'), out_channels)[1]

        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob, A):
        x = self.bn(self.conv(x))
        x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class unit_tcn_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn_skip, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = build_norm_layer(dict(type='BN'), out_channels)[1]

        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_weights()

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # N, R, V, V
        x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return x1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
            3, 1, num_point, num_point]), dtype=torch.float32, requires_grad=True).repeat(1, groups, 1, 1),
                                      requires_grad=True)
        # A: 3 27 27 -> reshape 3 1 27 27 -> repeat -> 3 16 27 27

        # self.A = nn.Parameter(torch.tensor(A.astype(np.float32)).clone())
        # self.alpha = nn.Parameter(torch.zeros(1))
        # self.convs = nn.ModuleList()
        # for i in range(self.num_subset):
        #     self.convs.append(CTRGC(in_channels, out_channels))
        #
        # self.ensemble = nn.Conv2d(out_channels * 2, out_channels, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                # build_norm_layer(dict(type='BN'), out_channels)[1]
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * num_subset)
        # self.bn0 = build_norm_layer(dict(type='BN'), out_channels * num_subset)[1]
        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = build_norm_layer(dict(type='BN'), out_channels)[1]

        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(
            in_channels, out_channels * num_subset, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(
            0.5 / (out_channels * num_subset)))

        self.Linear_bias = nn.Parameter(torch.zeros(
            1, out_channels * num_subset, 1, 1, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.constant(self.Linear_bias, 1e-6)

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(num_point))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(
            eye_array), requires_grad=False, device='cuda'), requires_grad=False)  # [c,25,25]

    def norm(self, A):
        b, c, h, w = A.size()
        A = A.view(c, self.num_point, self.num_point)
        D_list = torch.sum(A, 1).view(c, 1, self.num_point)
        D_list_12 = (D_list + 0.001) ** (-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0):
        learn_A = self.DecoupleA.repeat(
            1, self.out_channels // self.groups, 1, 1)
        norm_learn_A = torch.cat([self.norm(learn_A[0:1, ...]), self.norm(
            learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])], 0)

        x = torch.einsum(
            'nctw,cd->ndtw', (x0, self.Linear_weight)).contiguous()
        # x0 32 3 150 27 -> x 32 192 150 27
        x = x + self.Linear_bias
        x = self.bn0(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum('nkctv,kcvw->nctw', (x, norm_learn_A))
        # x 32 3 64 150 27 -> 3 64 150 27

        # y = None
        # for i in range(self.num_subset):
        #     z = self.convs[i](x0, self.A[i], self.alpha)
        #     y = z + y if y is not None else z
        #
        # x = torch.cat((x, y), dim=1)
        # x = self.ensemble(x)

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(-1)

    def forward(self, q, k, v):
        # q.size(): [nh*b x t x d_k]
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class LayerNormalization(nn.Module):
    """ Layer normalization module """

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, input_size, output_size, d_k, d_v, dropout=0.1, layer_norm=True):
        """
        Args:
            n_head: Number of attention heads
            input_size: Input feature size
            output_size: Output feature size
            d_k: Feature size for each head
            d_v: Feature size for each head
            dropout: Dropout rate after projection
        """
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, input_size, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, input_size, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, input_size, d_v))

        self.attention = ScaledDotProductAttention(input_size)
        self.layer_norm = LayerNormalization(input_size) if layer_norm else nn.Identity()

        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.w_qs)
        nn.init.xavier_normal_(self.w_ks)
        nn.init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)  # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)  # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)  # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        split_size = mb_size.item() if isinstance(mb_size, torch.Tensor) else mb_size
        h, t, e = outputs.size()
        outputs = outputs.view(h // split_size, split_size, t, e)  # (H x B x T x E)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(split_size, len_q, -1)  # (B x T x H*E)

        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_hid, d_inner_hid, dropout=0.1, layer_norm=True):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)  # position-wise
        self.layer_norm = LayerNormalization(d_hid) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class DecoderBlock(nn.Module):
    """ Compose with two layers """

    def __init__(self, input_size, hidden_size, inner_hidden_size, n_head, d_k, d_v, dropout=0.1, layer_norm=True):
        super(DecoderBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, input_size, hidden_size, d_k, d_v, dropout=dropout,
                                           layer_norm=layer_norm)
        self.pos_ffn = PositionwiseFeedForward(hidden_size, inner_hidden_size, dropout=dropout, layer_norm=layer_norm)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class PositionEncoding(nn.Module):
    def __init__(self, n_positions, hidden_size):
        super().__init__()
        self.enc = nn.Embedding(n_positions, hidden_size, padding_idx=0)

        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / hidden_size) for j in range(hidden_size)]
            if pos != 0 else np.zeros(hidden_size) for pos in range(n_positions)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        self.enc.weight = torch.nn.Parameter(torch.from_numpy(position_enc).to(self.enc.weight.device, torch.float))

    def forward(self, x):
        indeces = torch.arange(0, x.size(1)).to(self.enc.weight.device, torch.long)
        encodings = self.enc(indeces)
        x += encodings
        return x


class SelfAttention(nn.Module):
    """Process sequences using self attention."""

    def __init__(self, input_size, hidden_size, n_heads, sequence_size, inner_hidden_factor=2, layer_norm=True):
        super().__init__()

        input_sizes = [hidden_size] * len(n_heads)
        input_sizes[0] = input_size
        hidden_sizes = [hidden_size] * len(n_heads)

        # self.position_encoding = PositionEncoding(sequence_size, hidden_size)

        self.layers = nn.ModuleList([
            DecoderBlock(inp_size, hid_size, hid_size * inner_hidden_factor, n_head, hid_size // n_head,
                         hid_size // n_head, layer_norm=layer_norm)
            for i, (inp_size, hid_size, n_head) in enumerate(zip(input_sizes, hidden_sizes, n_heads))
        ])

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(N, T, C * V)
        outputs, attentions = [], []

        # x = self.position_encoding(x)

        for layer in self.layers:
            x, attn = layer(x)

            outputs.append(x)
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, block_size, stride=1, residual=True,
                 attention=True):
        super(TCN_GCN_unit, self).__init__()
        num_jpts = A.shape[-1]
        self.gcn1 = unit_gcn(in_channels, out_channels, A, groups, num_point)
        self.tcn1 = unit_tcn(out_channels, out_channels,
                             stride=stride, num_point=num_point)
        self.relu = nn.ReLU()

        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
            3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'),
                              requires_grad=False)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_skip(
                in_channels, out_channels, kernel_size=1, stride=stride)
        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)
        self.attention = attention
        if attention:
            print('Attention Enabled!')

            num_heads = 4
            num_layers = 2
            sequence_length = 100
            self.self_attention_decoder = SelfAttention(out_channels, out_channels,
                                                        [num_heads] * num_layers,
                                                        sequence_length, layer_norm=True)

            self.sigmoid = nn.Sigmoid()
            # temporal attention
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)
            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x, keep_prob):
        y = self.gcn1(x)
        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)

            y = self.self_attention_decoder(y)

        y = self.tcn1(y, keep_prob, self.A)
        x_skip = self.dropT_skip(self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        return self.relu(y + x_skip)


class unit_sgn(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        # x: N, C, T, V; A: N, T, V, V
        x1 = x.permute(0, 2, 3, 1).contiguous()
        x1 = A.matmul(x1).permute(0, 3, 1, 2).contiguous()
        return self.relu(self.bn(self.conv(x1) + self.residual(x)))


class SGN(nn.Module):

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_joints=25,
                 T=30,
                 bias=True):
        super(SGN, self).__init__()

        self.T = T
        self.num_joints = num_joints
        self.base_channel = base_channels

        self.joint_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.motion_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.t_embed = self.embed_mlp(self.T, base_channels * 4, base_channels, bias=bias)
        self.s_embed = self.embed_mlp(self.num_joints, base_channels, base_channels, bias=bias)
        self.joint_embed = self.embed_mlp(in_channels, base_channels, base_channels, bias=bias)
        self.motion_embed = self.embed_mlp(in_channels, base_channels, base_channels, bias=bias)

        self.compute_A1 = ConvModule(base_channels * 2, base_channels * 4, kernel_size=1, bias=bias)
        self.compute_A2 = ConvModule(base_channels * 2, base_channels * 4, kernel_size=1, bias=bias)

        self.tcn = nn.Sequential(
            # nn.AdaptiveMaxPool2d((100, 27)),
            ConvModule(base_channels * 4, base_channels * 4, kernel_size=(3, 1), padding=(1, 0), bias=bias,
                       norm_cfg=dict(type='BN2d')),
            nn.Dropout(0.2),
            ConvModule(base_channels * 4, base_channels, kernel_size=1, bias=bias, norm_cfg=dict(type='BN2d'))
        )

        self.gcn1 = unit_sgn(base_channels * 2, base_channels * 2, bias=bias)
        self.gcn2 = unit_sgn(base_channels * 2, base_channels * 4, bias=bias)
        self.gcn3 = unit_sgn(base_channels * 4, base_channels * 4, bias=bias)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.conv.weight, 0)
        nn.init.constant_(self.gcn2.conv.weight, 0)
        nn.init.constant_(self.gcn3.conv.weight, 0)

    def embed_mlp(self, in_channels, out_channels, mid_channels=64, bias=False):
        return nn.Sequential(
            ConvModule(in_channels, mid_channels, kernel_size=1, bias=bias),
            ConvModule(mid_channels, out_channels, kernel_size=1, bias=bias),
        )

    def compute_A(self, x):
        # X: N, C, T, V
        A1 = self.compute_A1(x).permute(0, 2, 3, 1).contiguous()
        A2 = self.compute_A2(x).permute(0, 2, 1, 3).contiguous()
        A = A1.matmul(A2)
        return nn.Softmax(dim=-1)(A)

    def forward(self, joint):
        N, M, T, V, C = joint.shape

        joint = joint.reshape(N * M, T, V, C)
        joint = joint.permute(0, 3, 2, 1).contiguous()
        # NM, C, V, T
        motion = torch.diff(joint, dim=3, append=torch.zeros(N * M, C, V, 1).to(joint.device))
        joint = self.joint_bn(joint.view(N * M, C * V, T))
        motion = self.motion_bn(motion.view(N * M, C * V, T))
        joint = joint.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
        motion = motion.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()

        joint_embed = self.joint_embed(joint)
        motion_embed = self.motion_embed(motion)
        # N * M, C, T, V
        t_code = torch.eye(T).to(joint.device)
        t_code = t_code[None, :, None].repeat(N * M, 1, V, 1)
        s_code = torch.eye(V).to(joint.device)
        s_code = s_code[None, ...,  None].repeat(N * M, 1, 1, T)
        t_embed = self.t_embed(t_code).permute(0, 1, 3, 2).contiguous()
        s_embed = self.s_embed(s_code).permute(0, 1, 3, 2).contiguous()

        x = torch.cat([joint_embed + motion_embed, s_embed], 1)
        # N * M, 2base, V, T
        A = self.compute_A(x)
        # N * M, T, V, V
        for gcn in [self.gcn1, self.gcn2, self.gcn3]:
            x = gcn(x, A)

        x = x + t_embed
        # x 32 256 100 27
        x = self.tcn(x)
        # N * M, C, T, V
        return x


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, groups=8, block_size=41, graph=None, graph_args=dict(),
                 in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        print(A)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # self.l0 = SGN(in_channels, 64, 27, 150, True)
        self.l1 = TCN_GCN_unit(in_channels, 64, A, groups, num_point,
                               block_size)
        self.l2 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.l3 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.l4 = TCN_GCN_unit(64, 64, A, groups, num_point, block_size)
        self.l5 = TCN_GCN_unit(
            64, 128, A, groups, num_point, block_size, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.l7 = TCN_GCN_unit(128, 128, A, groups, num_point, block_size)
        self.l8 = TCN_GCN_unit(128, 256, A, groups,
                               num_point, block_size, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)
        self.l10 = TCN_GCN_unit(256, 256, A, groups, num_point, block_size)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, keep_prob=0.9):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        # x = self.l0(x.reshape(N, M, T, V, C))

        x = self.l1(x, 1.0)

        x = self.l2(x, 1.0)
        x = self.l3(x, 1.0)
        x = self.l4(x, 1.0)
        x = self.l5(x, 1.0)
        x = self.l6(x, 1.0)
        x = self.l7(x, keep_prob)
        x = self.l8(x, keep_prob)
        x = self.l9(x, keep_prob)
        x = self.l10(x, keep_prob)

        # N*M,C,T,V
        c_new = x.size(1)

        # print(x.size())
        # print(N, M, c_new)

        # x = x.view(N, M, c_new, -1)
        x = x.reshape(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        # with open('feature.csv', 'a+') as t:
        #     wt = csv.writer(t, lineterminator='\n')
        #     for i in range(len(x)):
        #         wt.writerow(list([y.item() for y in x[i]]))
        return self.fc(x)
