# only for backup


# TRCSP layer  -----------------------------------------------------------------------------------
class MHSA_layer(nn.Module):
    # MHSA_layer layer
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA_layer, self).__init__()
        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]), requires_grad=True)
            self.rel_w = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        c1, c2, c3, c4 = content_content.size()
        if self.pos:

            content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3,
                                                                                                          2)  # 1,4,1024,64

            content_position = torch.matmul(content_position, q)
            content_position = content_position if (
                    content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
            assert (content_content.shape == content_position.shape)
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        return out


class BottleneckTransformer_1(nn.Module):
    # Bottleneck with MHSA_layer layer (shortcut fixed version)

    def __init__(self, c1, c2, stride=1, heads=4, mhsa=True, resolution=None, expansion=1, shortcut=True):
        super(BottleneckTransformer_1, self).__init__()
        c_ = int(c2 * expansion)
        self.cv1 = Conv(c1, c_, 1, 1)
        if not mhsa:
            self.cv2 = Conv(c_, c2, 3, 1)
        else:
            self.cv2 = nn.ModuleList()
            self.cv2.append(MHSA_layer(c2, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.cv2.append(nn.AvgPool2d(2, 2))
            self.cv2 = nn.Sequential(*self.cv2)

        self.resshortcut = shortcut and c1 == c2  # fix shortcut (res connection)

        if stride != 1 or c1 != expansion * c2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, expansion * c2, kernel_size=1, stride=stride),
                nn.BatchNorm2d(expansion * c2)
            )
        self.fc1 = nn.Linear(c2, c2)

    def forward(self, x):

        out = x + self.cv2(self.cv1(x)) if self.resshortcut else self.cv2(self.cv1(x))

        return out


class TRCSP_1(nn.Module):
    # CSP with MHSA_layer layer (shortcut fixed version)
    def __init__(self, c1, c2, n=1, w=20, h=20, shortcut=True, e=0.5, e2=1):  # ch_in, ch_out, number, , expansion,w,h
        super(TRCSP_1, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[
            BottleneckTransformer_1(c_, c_, stride=1, heads=4, mhsa=True, resolution=(w, h), expansion=e2,
                                    shortcut=shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


# coordinate attention  -----------------------------------------------------------------------------------
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# Weighted feature concat ------------------------------------------------------------------------------------------
class WF_Concat(nn.Module):
    def __init__(self, dimension=1):
        super(WF_Concat, self).__init__()
        self.d = dimension
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        if len(x) == 2:
            w = self.w1
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = [weight[0] * x[0], weight[1] * x[1]]
        elif len(x) == 3:
            w = self.w2
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]

        return torch.cat(x, self.d)


# AFPN -------------------------------------------------------------------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )


    def forward(self, x):
        x = self.upsample(x)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        x = self.downsample(x)

        return x


class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128]):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        self.upsample = Upsample(channel[1], channel[0])
        self.downsample = Downsample(channel[0], channel[1])
        self.level = level

    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)

        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return out


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[64, 128, 256]):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

        self.level = level
        if self.level == 0:
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
        elif self.level == 1:
            self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
        elif self.level == 2:
            self.downsample2x = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x = Downsample(channel[0], channel[2], scale_factor=4)

    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input1 = self.downsample2x1(input1)
        elif self.level == 2:
            input1 = self.downsample4x(input1)
            input2 = self.downsample2x(input2)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]

        out = self.conv(fused_out_reduced)

        return out