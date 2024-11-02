# PKCIn-Net w/ metric-based Adaptive Peak Convolution (AdaPKC-Xi)
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
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

# AdaPKC: PeakConv w/ Metric-based AdaPRF and Confidence threshold
class AdaPKC2D_Thre(nn.Module):
    def __init__(self, in_ch, out_ch, bias, refer_band, guard_bands_h, guard_bands_w, threshold=0., init_gb=(1, 1)):
        """
        Args:
            in_ch: the input channel.
            out_ch: the output channel.
            bias: the bias for peak convolution.
            refer_band: the reference bandwidth.
            guard_bands_h: the guard bandwidth search space in dim of h.
            guard_bands_w: the guard bandwidth search space in dim of w.
            threshold: confidence threshold.
            init_gb: initialized guard bandwidth.
        """
        super(AdaPKC2D_Thre, self).__init__()
        self.guard_bands_h = guard_bands_h
        self.guard_bands_w = guard_bands_w
        self.threshold = threshold
        self.init_gb = init_gb
        # padding order: l-r-t-b
        self.padding = (self.guard_bands_w[-1] + refer_band, 
                        self.guard_bands_w[-1] + refer_band,
                        self.guard_bands_h[-1] + refer_band,
                        self.guard_bands_h[-1] + refer_band)
        self.zero_padding = nn.ZeroPad2d(self.padding)
        self.refer_band = refer_band

        # kernel size of peak convolution
        self.kernel_size = (4, 4)
        self.stride = self.kernel_size
        self.bias = bias
        # stride of peak conv only supports 1 now
        self.pc_strd = 1
        # conv block for peak receptive field (PRF) in x
        # take AdaPRF as input and output the center point output of AdaPRF;
        # since each p_c is expanded as a tensor of kernel scale RF, the stride = kernel_size
        self.peak_conv = nn.Conv2d(in_ch, out_ch,
                                   kernel_size=self.kernel_size,
                                   padding=0,
                                   stride=self.stride,
                                   bias=self.bias)
    
    def forward(self, x):
        b, c, h, w = x.shape
        k_pk = self.kernel_size
        # prf size
        N = k_pk[0] * k_pk[1]
        dtype = x.data.type()

        x = self.zero_padding(x)
        # sample x_c (b, c, h, w, N) using p_c
        p_c = self._get_p_c(b, h, w, N, dtype)
        x_c = self._sample_x(x, p_c, N)

        # calculate metric scores for guard bandwidths
        mat_similiraty = []
        index_gb = 0
        for gb_h in self.guard_bands_h:
            for gb_w in self.guard_bands_w:
                # keep the index of initial guard bandwidth
                if gb_h == self.init_gb[0] and gb_w == self.init_gb[1]:
                    init_index = index_gb
                # acquire x_r corresponding to current guard bandwidth
                gb = (gb_h, gb_w)
                p_r = self._get_p_r(b, h, w, N, gb, dtype)
                # sample x_r (b, c, h, w, N) using p_r
                x_r = self._sample_x(x, p_r, N)
                # get metric score for current guard bandwidth
                # shape of (b, h, w)
                similiraty = torch.sigmoid(torch.mul(x_r, x_c).mean(dim=1)).mean(dim=-1)
                mat_similiraty.append(similiraty)
                index_gb += 1
        mat_similiraty = torch.stack(mat_similiraty, dim=-1)
        
        # calculate the maximum of the first order gradient
        # shape of (b, h, w, gb_h*gb_w)
        similiraty_sorted, sorted_index = torch.sort(mat_similiraty, dim=-1)
        # shape of (b, h, w)
        max_diff ,max_diff_index = torch.max(torch.diff(similiraty_sorted, dim=-1), dim=-1)
        # get the index of selected guard bandwidth, shape of (b, h, w)
        ori_index = torch.gather(sorted_index, dim=-1, index=max_diff_index.unsqueeze(-1)).squeeze(-1)
        # use confidence threshold to select proper guard bandwidth
        # for AdaPKC-Xi w/o FiTOS, threshold = 0.
        ori_index = ori_index * (max_diff >= self.threshold) + \
            init_index * torch.ones_like(ori_index) * (max_diff < self.threshold)
        
        # acquire adaprf for each x_c using selected guard bandwidth
        index_gb = 0
        pr_select = torch.zeros_like(p_c)
        for gb_h in self.guard_bands_h:
            for gb_w in self.guard_bands_w:
                # acquire p_r corresponding to current guard bandwidth
                gb = (gb_h, gb_w)
                p_r = self._get_p_r(b, h, w, N, gb, dtype)
                # shape of (b, 1, h, w)
                value_bool = (ori_index == index_gb).unsqueeze(1)
                pr_select.add_(p_r * value_bool)
                index_gb += 1
        xr_select = self._sample_x(x, pr_select, N)
        
        # getting x_prf for peak convolution
        x_prf = x_c - xr_select
        x_prf = self._reshape_x_prf(k_pk[0], k_pk[1], x_prf)

        # peak convolution
        out = self.peak_conv(x_prf)
        return out

    # getting p_c (the center point coords) from the padded grid of input x
    def _get_p_c(self, b, h, w, N, dtype):
        # generating pc_grid
        # the order of padding is l-r-t-b
        p_c_x, p_c_y = torch.meshgrid(
            torch.arange(self.padding[2], h*self.pc_strd+self.padding[2], self.pc_strd),
            torch.arange(self.padding[0], w*self.pc_strd+self.padding[0], self.pc_strd))
        p_c_x = torch.flatten(p_c_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_c_y = torch.flatten(p_c_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # p_c: (1, 2N, h, w)
        p_c = torch.cat([p_c_x, p_c_y], 1).type(dtype)
        # (b, 2N, h, w)
        p_c = p_c.repeat(b, 1, 1, 1)
        return p_c

    # generating peak receptive field grid with uniform sampling
    def _get_uni_prf_grid(self, rb, gb, N, dtype):
        # relative potision of receptive field grid
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
        # only sample 16 points, 5 for top and down directions, 3 for left and right directions
        index_td = torch.round(torch.linspace(0, (rb+gb[1])*2, 5)).long()
        index_lr = torch.round(torch.linspace(0, (rb+gb[0])*2, 5)).long()
        index_lr = index_lr[1:-1]

        prf_xt = prf_x_idx[0:rb, index_td]
        prf_xr = prf_x_idx[index_lr, (w_prf - rb):w_prf]
        prf_xd = prf_x_idx[(h_prf-rb):h_prf, index_td]
        prf_xl = prf_x_idx[index_lr, 0:rb]
        prf_x = torch.cat([torch.flatten(prf_xt),
                           torch.flatten(prf_xr),
                           torch.flatten(prf_xd),
                           torch.flatten(prf_xl)], 0)
        
        prf_yt = prf_y_idx[0:rb, index_td]
        prf_yr = prf_y_idx[index_lr, (w_prf - rb):w_prf]
        prf_yd = prf_y_idx[(h_prf-rb):h_prf, index_td]
        prf_yl = prf_y_idx[index_lr, 0:rb]
        prf_y = torch.cat([torch.flatten(prf_yt),
                           torch.flatten(prf_yr),
                           torch.flatten(prf_yd),
                           torch.flatten(prf_yl)], 0)

        prf = torch.cat([prf_x, prf_y], 0)
        prf = prf.view(1, 2*N, 1, 1).type(dtype)
        return prf


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
    def _get_p_r(self, b, h, w, N, gb, dtype):
        rb = self.refer_band
        # (1, 2N, 1, 1)
        prf = self._get_uni_prf_grid(rb, gb, N, dtype)
        # (B, 2N, h, w)
        p_c = self._get_p_c(b, h, w, N, dtype)
        # (B, 2N, h, w)
        p_r = p_c + prf
        return p_r
    
    # sampling x using p_r or p_c
    def _sample_x(self, x_pad, p, N):
        # (b, 2N, h, w) -> (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1).floor()
        b, h, w, _ = p.size()
        # x_pad: shape of (b, c, h_pad, w_pad)
        h_pad = x_pad.size(2)
        w_pad = x_pad.size(3)
        c = x_pad.size(1)
        # strech each spatial channel of x_pad as 1-D vector
        # shape of (b, c, h_pad*w_pad)
        x_pad = x_pad.contiguous().view(b, c, -1)
        # (b, h, w, N)
        # transform spatial coord of p into the 1-D index
        index = p[..., :N]*w_pad + p[..., N:]
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_r = x_pad.gather(dim=-1, index=index.long()).contiguous().view(b, c, h, w, N)
        return x_r

    @staticmethod
    # reshape the x_prf
    def _reshape_x_prf(k_h, k_w, x_prf):
        b, c, h, w, N = x_prf.size()
        x_prf = torch.cat([x_prf[..., s:s+k_w].contiguous().view(b, c, h, w*k_w) for s in range(0, N, k_w)], dim=-1)
        x_prf = x_prf.contiguous().view(b, c, h*k_h, w*k_w)
        return x_prf

# Double Adaptive PeakConv 2D Block
class DoubleAdaPKC2D(nn.Module):
    """ (2D AdaPKC => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, bias, rb, sig, thre=0.):
        super(DoubleAdaPKC2D, self).__init__()
        self.bias = bias
        self.refer_band = rb
        self.signal_type = sig
        # initialized guard bandwidth
        self.init_gb = (1, 1)
        # Confidence threshold
        # 1. For original adapkc xi, confidence threshold = 0.
        # 2. For original pkcin, confidence threshold = 1.
        # 3. For adapkc xi w/ fitos, default confidence threshold = 0.6
        self.threshold = thre
        # Search space of guard bandwidth
        if self.signal_type == 'range_doppler':
            self.gb_h = [1, 2]
            self.gb_w = [1, 2, 3]
        elif self.signal_type == 'angle_doppler':
            self.gb_h = [1, 2]
            self.gb_w = [1, 2, 3]
        # range_angle
        elif self.signal_type == 'range_angle':
            self.gb_h = [1, 2]
            self.gb_w = [1, 2, 3]
        else:
            raise KeyError('signal type is not correct: {}'.format(self.signal_type))
        self.pk_conv1 = AdaPKC2D_Thre(in_ch, out_ch, bias=self.bias,
                                   refer_band=self.refer_band,
                                   guard_bands_h=self.gb_h,
                                   guard_bands_w=self.gb_w,
                                   init_gb=self.init_gb,
                                   threshold=self.threshold)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.pk_conv2 = AdaPKC2D_Thre(out_ch, out_ch, bias=self.bias,
                                   refer_band=self.refer_band,
                                   guard_bands_h=self.gb_h,
                                   guard_bands_w=self.gb_w,
                                   init_gb=self.init_gb,
                                   threshold=self.threshold)
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
    threshold: confidence threshold in 0~1.
    """

    def __init__(self, signal_type, threshold=0.):
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
                                                 thre=threshold)
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


class AdaPKC_Xi(nn.Module):
    """ 
    Metric-based Adaptive Peak Convolution RSS Network (AdaPKC-Xi)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    threshold: float
        Confidence threshold
    """

    def __init__(self, n_classes, n_frames, threshold=0.):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler', threshold=threshold)
        self.ra_encoding_branch = EncodingBranch('range_angle', threshold=threshold)
        self.ad_encoding_branch = EncodingBranch('angle_doppler', threshold=threshold)

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
