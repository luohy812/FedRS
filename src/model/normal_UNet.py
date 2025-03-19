import torch
import torch.nn as nn
import torch.nn.functional as F


# ConvBlock
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),  # 输入通道，输出通道，卷积核大小，填充操作，边缘添加一圈的填充，保持特征图大小与输入相同
            nn.InstanceNorm3d(out_ch),  # 标准化特征图
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),  # 对outch再次卷积
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):  # 把输入张量通过conv进行前向转播，得到输出特征图y
        y = self.conv(x)
        return y


# Encoding block
class enc_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(enc_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)  # 把经过double_conv输出的特征图给到conv
        self.down = nn.MaxPool3d(2)  # 3d最大池化

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.down(y_conv)  # 对其池化操作
        return y, y_conv


# Decoding block
class dec_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dec_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.up = nn.ConvTranspose3d(out_ch, out_ch, 2, stride=2)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.up(y_conv)  # 卷积完上采样
        return y, y_conv


def concatenate(x1, x2):  # 连接
    diffZ = x2.size()[2] - x1.size()[2]  # 计算x1和x2的差距
    diffY = x2.size()[3] - x1.size()[3]
    diffX = x2.size()[4] - x1.size()[4]
    x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,  # 除2取整
                    diffY // 2, diffY - diffY // 2,
                    diffZ // 2, diffZ - diffZ // 2))
    y = torch.cat([x2, x1], dim=1)  # 将x1 x2连接
    return y


class sub_encoder(nn.Module):
    def __init__(self, in_ch, base_ch):
        super(sub_encoder, self).__init__()  # 编码器
        self.enc1 = enc_block(in_ch, base_ch)
        self.enc2 = enc_block(base_ch, base_ch * 2)  # 逐渐增加通道数
        self.enc3 = enc_block(base_ch * 2, base_ch * 4)
        self.enc4 = enc_block(base_ch * 4, base_ch * 8)

    def forward(self, x):
        y, enc_conv_1 = self.enc1(x)
        y, enc_conv_2 = self.enc2(y)
        y, enc_conv_3 = self.enc3(y)
        y, enc_conv_4 = self.enc4(y)
        return y, enc_conv_1, enc_conv_2, enc_conv_3, enc_conv_4


class sub_decoder(nn.Module):
    def __init__(self, base_ch, cls_num):
        super(sub_decoder, self).__init__()
        self.dec1 = dec_block(base_ch * 8, base_ch * 8)
        self.dec2 = dec_block(base_ch * 16, base_ch * 4)  # 逐渐减少通道数
        self.dec3 = dec_block(base_ch * 8, base_ch * 2)
        self.dec4 = dec_block(base_ch * 4, base_ch)
        self.lastconv = double_conv(base_ch * 2, base_ch)
        self.outconv = nn.Conv3d(base_ch, cls_num + 1, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, e1, e2, e3, e4):
        y, ds1 = self.dec1(x)

        y, ds2 = self.dec2(concatenate(y, e4))

        y, ds3 = self.dec3(concatenate(y, e3))

        y, ds4 = self.dec4(concatenate(y, e2))

        y = self.lastconv(concatenate(y, e1))

        y = self.outconv(y)

        output = self.softmax(y)

        return output

class SEAttention3D(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 3D自适应平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()  # 3D输入大小 (batch, channel, depth, height, width)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)  # 3D形状
        return x * y.expand_as(x)
class sub_decoder_attention(nn.Module):
    def __init__(self, base_ch, cls_num, attention_name="SE"):
        super(sub_decoder_attention, self).__init__()
        self.dec1 = dec_block(base_ch * 8, base_ch * 8)
        self.dec2 = dec_block(base_ch * 16, base_ch * 4)  # 逐渐减少通道数
        self.dec3 = dec_block(base_ch * 8, base_ch * 2)
        self.dec4 = dec_block(base_ch * 4, base_ch)
        self.lastconv = double_conv(base_ch * 2, base_ch)
        self.outconv = nn.Conv3d(base_ch, cls_num + 1, 1)
        self.softmax = nn.Softmax(dim=1)
        self.attention_name = attention_name


        self.att1 = SEAttention3D(base_ch * 8)
        self.att2 = SEAttention3D(base_ch * 4)
        self.att3 = SEAttention3D(base_ch * 2)
        self.att4 = SEAttention3D(base_ch)
    def forward(self, x, e1, e2, e3, e4):
        y, ds1 = self.dec1(x)
        y = self.att1(y)

        y, ds2 = self.dec2(concatenate(y, e4))
        y = self.att2(y)

        y, ds3 = self.dec3(concatenate(y, e3))
        y = self.att3(y)

        y, ds4 = self.dec4(concatenate(y, e2))
        y = self.att4(y)

        y = self.lastconv(concatenate(y, e1))

        y = self.outconv(y)

        output = self.softmax(y)

        return output

class sub_decoder_proto(nn.Module):
    def __init__(self, base_ch, cls_num):
        super(sub_decoder_proto, self).__init__()
        self.dec1 = dec_block(base_ch * 8, base_ch * 8)
        self.dec2 = dec_block(base_ch * 16, base_ch * 4)  # 逐渐减少通道数
        self.dec3 = dec_block(base_ch * 8, base_ch * 2)
        self.dec4 = dec_block(base_ch * 4, base_ch)
        self.lastconv = double_conv(base_ch * 2, base_ch)
        self.outconv = nn.Conv3d(base_ch, cls_num + 1, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, e1, e2, e3, e4):
        y, ds1 = self.dec1(x)

        y, ds2 = self.dec2(concatenate(y, e4))

        y, ds3 = self.dec3(concatenate(y, e3))

        y, ds4 = self.dec4(concatenate(y, e2))

        y = self.lastconv(concatenate(y, e1))
        y = self.outconv(y)
        output = self.softmax(y)

        return output, y


class UNet_proto(nn.Module):
    def __init__(self, in_ch, base_ch, cls_num):
        super(UNet_proto, self).__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.cls_num = cls_num
        self.local_encoders = sub_encoder(in_ch, base_ch)
        # self.sub_encoders = sub_encoder(in_ch, base_ch)

        self.global_decoder = sub_decoder_proto(base_ch, cls_num)

    def forward(self, x, node_enabled_encoders=None):
        y, e1, e2, e3, e4 = self.local_encoders(x)
        output, proto = self.global_decoder(y, e1, e2, e3, e4)
        return output, proto

    def get_s1_parameters(self):
        params = list(self.global_decoder.parameters())
        params += list(self.local_encoders.parameters())
        return params

    def get_s2_parameters(self, node_enabled_encoders):
        params = list(self.aux_decoder.parameters())
        for encoder_id in node_enabled_encoders:
            params += list(self.sub_encoders[encoder_id].parameters())
        return params

    def description(self):
        return 'Multi-encoder U-Net (input channel = {0:d}) for {1:d}-organ segmentation (base channel = {2:d})'.format(
            self.in_ch, self.cls_num, self.base_ch)


class Unet(nn.Module):
    def __init__(self, in_ch, base_ch, cls_num):
        super(Unet, self).__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.cls_num = cls_num
        self.local_encoders = sub_encoder(in_ch, base_ch)
        # self.sub_encoders = sub_encoder(in_ch, base_ch)

        self.global_decoder = sub_decoder(base_ch, cls_num)

    def forward(self, x, node_enabled_encoders=None):
        y, e1, e2, e3, e4 = self.local_encoders(x)
        output = self.global_decoder(y, e1, e2, e3, e4)

        return output

    def get_s1_parameters(self):
        params = list(self.global_decoder.parameters())
        params += list(self.local_encoders.parameters())
        return params


    def description(self):
        return 'Multi-encoder U-Net (input channel = {0:d}) for {1:d}-organ segmentation (base channel = {2:d})'.format(
            self.in_ch, self.cls_num, self.base_ch)
    def parameter_count(self, module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)


class Unet_attention(nn.Module):
    def __init__(self, in_ch, base_ch, cls_num, attention_name="SE"):
        super(Unet_attention, self).__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.cls_num = cls_num
        self.local_encoders = sub_encoder(in_ch, base_ch)
        self.attention_name = attention_name
        # self.sub_encoders = sub_encoder(in_ch, base_ch)

        self.global_decoder = sub_decoder_attention(base_ch, cls_num, attention_name)

    def forward(self, x, node_enabled_encoders=None):
        y, e1, e2, e3, e4 = self.local_encoders(x)
        output = self.global_decoder(y, e1, e2, e3, e4)

        return output

    def get_s1_parameters(self):
        params = list(self.global_decoder.parameters())
        params += list(self.local_encoders.parameters())
        return params


    def description(self):
        return 'Multi-encoder U-Net (input channel = {0:d}) for {1:d}-organ segmentation (base channel = {2:d})'.format(
            self.in_ch, self.cls_num, self.base_ch)



# model = Unet(in_ch=1, base_ch=20, cls_num=5)
#
# sub_encoder_params = model.parameter_count(model.local_encoders) / 1e6  # 单个子编码器参数量 (百万)
#
# global_decoder_params = model.parameter_count(model.global_decoder) / 1e6 # 共享解码器参数量 (百万)
#
# print(f"local_encoders parameters: {sub_encoder_params:.2f}M")
# print(f"Global decoder parameters: {global_decoder_params:.2f}M")