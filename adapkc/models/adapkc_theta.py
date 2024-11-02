# PKCIn-Net w/ learning-based Adaptive Peak Convolution (AdaPKC-Theta)
# The implementation of Adaptive Peak Convolution referred deformable Conv (https://github.com/4uiiurz1/pytorch-deform-conv-v2)
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Double3DConvBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


# estimation network for guard bandwidth
class BandEst(nn.Module):
    """
    Description:
        *Guard band estimator, BandEst
        *Vertical 1D Conv: (gb_max * 2 + 1, 1);
        *Horizontal 1D Conv: (1, gb_max * 2 + 1);
        input: x in the shape of BxCxHxW
        output: gb_mask in the shape of Bx4xHxW
    """
    def __init__(self, in_ch, gb_max):
        super(BandEst, self).__init__()
        self.in_ch = in_ch
        self.gb_max = gb_max
        self.single_conv_vrt = nn.Conv2d(self.in_ch,  2, kernel_size=(gb_max*2+1, 1), padding=(gb_max, 0), dilation=1)
        self.bn_vrt = nn.BatchNorm2d(2)
        self.single_conv_hrz = nn.Conv2d(self.in_ch,  2, kernel_size=(1, gb_max*2+1), padding=(0, gb_max), dilation=1)
        self.bn_hrz = nn.BatchNorm2d(2)
        # weight initialization
        nn.init.constant_(self.single_conv_vrt.weight, 0)
        nn.init.constant_(self.single_conv_hrz.weight, 0)

    def forward(self, x):
        # (b, 2, h, w)
        gb_map_v = self.bn_vrt(self.single_conv_vrt(x))
        # (b, 2, h, w)
        gb_map_h = self.bn_hrz(self.single_conv_hrz(x))
        # modulate to (0, self.gb_max - 1).
        gb_map_v = torch.sigmoid(gb_map_v) * (self.gb_max - 1)
        gb_map_h = torch.sigmoid(gb_map_h) * (self.gb_max - 1)
        # (b, 4, h, w)
        # in the direction of vt-vb-hl-hr
        return torch.cat((gb_map_v, gb_map_h), 1)

# AdaPKC: PeakConv w/ Learing-based AdaPRF
class AdaPeakConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, bias, refer_band, signal_type, gb_max=3):
        """
        Args:
            in_ch: the input channel.
            out_ch: the output channel.
            bias: the bias for peak convolution.
            refer_band: the reference bandwidth.
            signal_type: str in ('range_doppler', 'angle_doppler', 'range_angle').
            gb_max: the upper bound of guard bandwidth in four directions.
        """
        super(AdaPeakConv2D, self).__init__()
        self.signal_type = signal_type
        self.in_ch = in_ch
        self.out_ch = out_ch
        # the initialized guard bandwidth, also the lower bound of guard bandwidth
        self.init_guard_band = (1, 1)
        # padding order: l-r-t-b
        self.padding = (self.init_guard_band[1] + refer_band,
                        self.init_guard_band[1] + refer_band,
                        self.init_guard_band[0] + refer_band,
                        self.init_guard_band[0] + refer_band)
        self.refer_band = refer_band
        # stride of peak conv only supports 1 now
        self.pc_strd = 1
        self.rep_padding = nn.ReplicationPad2d(self.padding)
        # the estimation network of guard bandwidth
        self.gb_estimator = BandEst(self.in_ch, gb_max)
        
        # conv block for peak receptive field (PRF) in x
        self.kernel_size = (self.init_guard_band[0]+self.init_guard_band[1]+self.refer_band*2), 4
        self.stride = self.kernel_size
        self.bias = bias
        self.peak_conv = nn.Conv2d(self.in_ch, self.out_ch,
                                   kernel_size=self.kernel_size,
                                   padding=0,
                                   stride=self.stride,
                                   bias=self.bias)

    # update p_r with initialized p_r and estimated guard bandwidth offsets
    def get_pr_learned(self, pr, gb_mask):
        b, N, h, w = pr.size()
        gb_x_t = - gb_mask[:, 0, :, :].view(b, 1, h, w)
        gb_x_b = gb_mask[:, 1, :, :].view(b, 1, h, w)
        gb_y_l = - gb_mask[:, 2, :, :].view(b, 1, h, w)
        gb_y_r = gb_mask[:, 3, :, :].view(b, 1, h, w)
        zero = torch.zeros([int(b), int(N/8), int(h), int(w)], device=pr.device)
        move = torch.cat([gb_x_t.repeat(1, int(N/8), 1, 1), zero,
                          gb_x_b.repeat(1, int(N/8), 1, 1), zero,
                          zero, gb_y_r.repeat(1, int(N/8), 1, 1),
                          zero, gb_y_l.repeat(1, int(N/8), 1, 1)], 1)
        pr = pr + move
        return pr

    def forward(self, x):
        b, _, h, w = x.size()
        # kernel_size for pkc
        k_pkc = self.kernel_size
        # prf size
        N = k_pkc[0] * k_pkc[1]
        dtype = x.data.type()

        x_pad = self.rep_padding(x)
        # p_c denotes the center point positions
        p_c = self._get_p_c(b, h, w, N, dtype)
        # p_r denotes the initialized reference point positions
        # p_r shape: Bx2NxHxW
        p_r = self._get_p_r(N, p_c, dtype)
        # sample x_c (b,c,h,w,N) using p_c
        # (b, 2N, h, w) -> (b, h, w, 2N)
        p_c = p_c.contiguous().permute(0, 2, 3, 1).floor().long()
        x_c = self._sample_x(x_pad, p_c, N)

        # move initialized p_r with estimated gb_mask
        # estimate guard bandwidth offsets.
        # shape pf (b, 4, h, w)
        gb_mask = self.gb_estimator(x)
        # update p_r with offsets and initialized p_r
        # shape of (b, 2N, h, w)
        p = self.get_pr_learned(p_r, gb_mask)

        # put channel dimension at the last
        # shape of (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # let linear interpolation operate on q = [q_lt, q_rb] for each p
        # q in left/top position is the nearest integer number of p
        q_lt = p.detach().floor()
        # q in right/below position = 1 + q in left/top position
        q_rb = q_lt + 1

        # restrict q in the range of [(0, 0), (H-1, W-1)], the H and W is the shape of padded x.
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x_pad.size(2)-1), torch.clamp(q_lt[..., N:], 0, x_pad.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x_pad.size(2)-1), torch.clamp(q_rb[..., N:], 0, x_pad.size(3)-1)], dim=-1).long()

        # restrict p in the range of [(0, 0), (H-1, W-1)]
        p = torch.cat([torch.clamp(p[..., :N], 0, x_pad.size(2)-1), torch.clamp(p[..., N:], 0, x_pad.size(3)-1)], dim=-1)

        # using q to get linear kernel weights for p
        # shape of (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))

        # getting x(q) from input x
        # shape of (b, c, h, w, N)
        x_q_lt = self._sample_x(x_pad, q_lt, N)
        x_q_rb = self._sample_x(x_pad, q_rb, N)

        # weighted summation of linear kernel interpolation
        # (b, c, h, w, N)
        x_r = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb

        # getting x_prf for peak convolution
        x_prf = x_c - x_r
        x_prf = self._reshape_x_prf(k_pkc[0], k_pkc[1], x_prf)

        # peak convolution
        out = self.peak_conv(x_prf)
        return out

    # getting p_c (the center point coords) from the padded grid of input x
    def _get_p_c(self, b, h, w, N, dtype):
        # generating pc_grid
        p_c_x, p_c_y = torch.meshgrid(
            torch.arange(self.padding[2], h * self.pc_strd + self.padding[2], self.pc_strd),
            torch.arange(self.padding[0], w * self.pc_strd + self.padding[0], self.pc_strd))
        p_c_x = torch.flatten(p_c_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_c_y = torch.flatten(p_c_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # p_c: 1x2NxHxW
        p_c = torch.cat([p_c_x, p_c_y], 1).type(dtype)
        # (b,2N,h,w)
        p_c = p_c.repeat(b, 1, 1, 1)
        return p_c

    # generating peak receptive field grid
    def _gen_prf_grid(self, rb, gb, N, dtype):
        # h for row (x); w for col (y)
        h_t = -(rb + gb[0])
        h_d = rb + gb[0]
        w_l = -(rb + gb[1])
        w_r = rb + gb[1]
        # width and height of receptive field
        w_prf = (rb + gb[1]) * 2 + 1
        h_prf = (rb + gb[0]) * 2 + 1

        prf_x_idx, prf_y_idx = torch.meshgrid(
            torch.arange(h_t, h_d + 1),
            torch.arange(w_l, w_r + 1))

        # taking positions clockwise 
        prf_xt = prf_x_idx[0:rb, 0:(w_prf - rb)]
        prf_xr = prf_x_idx[0:(h_prf - rb), (w_prf - rb):w_prf]
        prf_xd = prf_x_idx[(h_prf - rb):h_prf, rb:w_prf]
        prf_xl = prf_x_idx[rb:h_prf, 0:rb]

        prf_x = torch.cat([torch.flatten(prf_xt),
                           torch.flatten(prf_xr),
                           torch.flatten(prf_xd),
                           torch.flatten(prf_xl)], 0)

        prf_yt = prf_y_idx[0:rb, 0:(w_prf - rb)]
        prf_yr = prf_y_idx[0:(h_prf - rb), (w_prf - rb):w_prf]
        prf_yd = prf_y_idx[(h_prf - rb):h_prf, rb:w_prf]
        prf_yl = prf_y_idx[rb:h_prf, 0:rb]

        prf_y = torch.cat([torch.flatten(prf_yt),
                           torch.flatten(prf_yr),
                           torch.flatten(prf_yd),
                           torch.flatten(prf_yl)], 0)

        prf = torch.cat([prf_x, prf_y], 0)
        prf = prf.view(1, 2 * N, 1, 1).type(dtype)
        return prf

    # getting p_r positions from each p_c
    def _get_p_r(self, N, p_c, dtype):
        # (1, 2N, 1, 1)
        prf = self._gen_prf_grid(self.refer_band, self.init_guard_band, N, dtype)
        # (B, 2N, h, w)
        p_r = p_c + prf
        return p_r

    # sampling x using p_r or p_c
    def _sample_x(self, x_pad, p, N):
        b, h, w, _ = p.size()
        # x_pad: shape of (b, c, h_pad, w_pad)
        h_pad = x_pad.size(2)
        w_pad = x_pad.size(3)
        c = x_pad.size(1)
        # strech each spatial channel of x_pad as 1-D vector
        x_pad = x_pad.contiguous().view(b, c, -1)
        # transform spatial coord of p into the 1-D index
        index = p[..., :N] * w_pad + p[..., N:]
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_r = x_pad.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_r

    @staticmethod
    # reshape the x_prf
    def _reshape_x_prf(k_h, k_w, x_prf):
        b, c, h, w, N = x_prf.size()
        x_prf = torch.cat([x_prf[..., s:s + k_w].contiguous().view(b, c, h, w * k_w) for s in range(0, N, k_w)], dim=-1)
        x_prf = x_prf.contiguous().view(b, c, h * k_h, w * k_w)
        return x_prf


# Double Adaptive PeakConv 2D Block
class DoubleAdaPKC2D(nn.Module):
    """ (2D AdaPKC => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, bias, rb, sig, gb_max=3):
        super(DoubleAdaPKC2D, self).__init__()
        self.bias = bias
        self.refer_band = rb
        self.gb_max = gb_max
        self.signal_type = sig
        self.pk_conv1 = AdaPeakConv2D(in_ch, out_ch, bias=self.bias,
                                      refer_band=self.refer_band,
                                      signal_type=self.signal_type,
                                      gb_max=self.gb_max)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.pk_conv2 = AdaPeakConv2D(out_ch, out_ch, bias=self.bias,
                                      refer_band=self.refer_band,
                                      signal_type=self.signal_type,
                                      gb_max=self.gb_max)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.pk_conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pk_conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_avg_pool = nn.AvgPool2d((64, 64))
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        x1 = F.interpolate(self.global_avg_pool(x), size=(64, 64), align_corners=False,
                           mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat


class EncodingBranch(nn.Module):
    """
    Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    """

    def __init__(self, signal_type):
        super().__init__()
        self.signal_type = signal_type
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=128, k_size=3,
                                                      pad=(0, 1, 1), dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.double_adapkc_block = DoubleAdaPKC2D(in_ch=128, out_ch=128,
                                                 bias=True,
                                                 rb=1,
                                                 sig=self.signal_type,
                                                 gb_max=3)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1,
                                                pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)  # remove temporal dimension

        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x1_pad = F.pad(x1, (0, 1, 0, 0), "constant", 0)
            x1_down = self.doppler_max_pool(x1_pad)
        else:
            x1_down = self.max_pool(x1)

        x2 = self.double_adapkc_block(x1_down)
        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x2_pad = F.pad(x2, (0, 1, 0, 0), "constant", 0)
            x2_down = self.doppler_max_pool(x2_pad)
        else:
            x2_down = self.max_pool(x2)

        x3 = self.single_conv_block1_1x1(x2_down)
        # return input of ASPP block + latent features
        return x2_down, x3


class AdaPKC_Theta(nn.Module):
    """ 
    Learning-based Adaptive Peak Convolution RSS Network (AdaPKC-Theta)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler')
        self.ra_encoding_branch = EncodingBranch('range_angle')
        self.ad_encoding_branch = EncodingBranch('angle_doppler')

        # ASPP Blocks
        self.rd_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ra_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ad_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ad_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

        # Decoding
        self.rd_single_conv_block2_1x1 = ConvBlock(in_ch=384, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block2_1x1 = ConvBlock(in_ch=384, out_ch=128, k_size=1, pad=0, dil=1)

        # Pallel range-Doppler (RD) and range-angle (RA) decoding branches
        self.rd_upconv1 = nn.ConvTranspose2d(384, 128, (2, 1), stride=(2, 1))
        self.ra_upconv1 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        self.ra_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)

        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)
        self.ra_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def forward(self, x_rd, x_ra, x_ad):
        # Backbone
        ra_features, ra_latent = self.ra_encoding_branch(x_ra)
        rd_features, rd_latent = self.rd_encoding_branch(x_rd)
        ad_features, ad_latent = self.ad_encoding_branch(x_ad)

        # ASPP blocks
        x1_rd = self.rd_aspp_block(rd_features)
        x1_ra = self.ra_aspp_block(ra_features)
        x1_ad = self.ad_aspp_block(ad_features)
        x2_rd = self.rd_single_conv_block1_1x1(x1_rd)
        x2_ra = self.ra_single_conv_block1_1x1(x1_ra)
        x2_ad = self.ad_single_conv_block1_1x1(x1_ad)

        # Latent Space
        # Features join either the RD or the RA branch
        x3 = torch.cat((rd_latent, ra_latent, ad_latent), 1)
        x3_rd = self.rd_single_conv_block2_1x1(x3)
        x3_ra = self.ra_single_conv_block2_1x1(x3)

        # Latent Space + ASPP features
        x4_rd = torch.cat((x2_rd, x3_rd, x2_ad), 1)
        x4_ra = torch.cat((x2_ra, x3_ra, x2_ad), 1)

        # Parallel decoding branches with upconvs
        x5_rd = self.rd_upconv1(x4_rd)
        x5_ra = self.ra_upconv1(x4_ra)
        x6_rd = self.rd_double_conv_block1(x5_rd)
        x6_ra = self.ra_double_conv_block1(x5_ra)

        x7_rd = self.rd_upconv2(x6_rd)
        x7_ra = self.ra_upconv2(x6_ra)
        x8_rd = self.rd_double_conv_block2(x7_rd)
        x8_ra = self.ra_double_conv_block2(x7_ra)

        # Final 1D convolutions
        x9_rd = self.rd_final(x8_rd)
        x9_ra = self.ra_final(x8_ra)

        return x9_rd, x9_ra