# PKCIn-Net with different sampling strategies
# The implementation of Peak Conv referred deformable Conv (https://github.com/4uiiurz1/pytorch-deform-conv-v2)
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

# Peak Conv with adaptive sampling strageties for reference units
class PKC2D_Sample(nn.Module):
    def __init__(self, in_ch, out_ch, bias, refer_band, guard_band_h, guard_band_w, samping_type='v1'):
        """
        Args:
            in_ch: the input channel.
            out_ch: the output channel.
            bias: the bias for peak convolution.
            refer_band: the reference bandwidth.
            guard_band_h: the guard bandwidth in dim of h.
            guard_band_w: the guard bandwidth in dim of w.
        """
        super(PKC2D_Sample, self).__init__()
        self.sampling_type = samping_type
        self.guard_band_h = guard_band_h
        self.guard_band_w = guard_band_w
        # padding order: l-r-t-b
        self.padding = (self.guard_band_w + refer_band, 
                        self.guard_band_w + refer_band,
                        self.guard_band_h + refer_band,
                        self.guard_band_h + refer_band)
        self.zero_padding = nn.ZeroPad2d(self.padding)
        self.refer_band = refer_band

        # kernel size of peak convolution
        self.kernel_size = (4, 4)
        self.stride = self.kernel_size
        self.bias = bias
        # stride of peak conv only supports 1 now
        self.pc_strd = 1
        # conv block for peak receptive field (PRF) in x
        # take PRF as input and output the center point output of PRF;
        # kernel channel = in_ch*2: depth_concat(p_c, p_r)
        # since each p_c is expanded as a tensor of kernel scale RF, the stride = kernel_size
        self.peak_conv = nn.Conv2d(in_ch, out_ch,
                                   kernel_size=self.kernel_size,
                                   padding=0,
                                   stride=self.stride,
                                   bias=self.bias)
    
    def forward(self, x):
        # shape of x is (b, c, h, w)
        b, c, h, w = x.shape
        k_pk = self.kernel_size
        # prf size
        N = k_pk[0] * k_pk[1]
        dtype = x.data.type()

        x = self.zero_padding(x)

        gb = (self.guard_band_h, self.guard_band_w)
        # number of overall units in reference band
        N_all = 4 * self.refer_band * (1 + self.refer_band + gb[0] + gb[1])

        # sample xc_all (b, c, h, w, N_all) using pc_all
        pc_all = self._get_p_c(b, h, w, N_all, dtype)
        xc_all = self._sample_x(x, pc_all, N_all)

        # let pr_all denote the reference point positions
        # pr_all shape: (b, 2*N_all, h, w)
        pr_all = self._get_p_r(b, h, w, N_all, gb, dtype)
        # sample xr_all (b, c, h, w, N_all) using pr_all
        xr_all = self._sample_x(x, pr_all, N_all)

        # calculate metric scores for reference units with center unit
        # shape of (b, h, w, N_all)
        similiraty = torch.sigmoid(torch.mul(xr_all, xc_all).mean(dim=1))
        _, sorted_index = torch.sort(similiraty, dim=-1)
        # get the index of selected reference units, shape of (b, h, w, N)
        if self.sampling_type == 'v1':
            # select the N least similar reference units
            ori_index = sorted_index[..., 0:N].unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous()
        else:
            # select the N reference units with intermediate similarity
            ori_index = sorted_index[..., int((N_all-N)/2):int((N_all+N)/2)].unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous()
        
        # resample reference units
        x_r = torch.gather(xr_all, dim=-1, index=ori_index)

        # sample x_c (b, c, h, w, N) using p_c
        p_c = self._get_p_c(b, h, w, N, dtype)
        x_c = self._sample_x(x, p_c, N)

        # getting x_prf for peak convolution
        x_prf = x_c - x_r
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

    # generating peak receptive field grid
    def _gen_prf_grid(self, rb, gb, N, dtype):
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
        prf_xt = prf_x_idx[0:rb]
        prf_xr = prf_x_idx[rb:(h_prf - rb), (w_prf - rb):w_prf]
        prf_xd = prf_x_idx[(h_prf-rb):h_prf]
        prf_xl = prf_x_idx[rb:(h_prf - rb), 0:rb]

        prf_x = torch.cat([torch.flatten(prf_xt),
                           torch.flatten(prf_xr),
                           torch.flatten(prf_xd),
                           torch.flatten(prf_xl)], 0)

        prf_yt = prf_y_idx[0:rb]
        prf_yr = prf_y_idx[rb:(h_prf - rb), (w_prf - rb):w_prf]
        prf_yd = prf_y_idx[(h_prf-rb):h_prf]
        prf_yl = prf_y_idx[rb:(h_prf - rb), 0:rb]

        prf_y = torch.cat([torch.flatten(prf_yt),
                           torch.flatten(prf_yr),
                           torch.flatten(prf_yd),
                           torch.flatten(prf_yl)], 0)

        prf = torch.cat([prf_x, prf_y], 0)
        prf = prf.view(1, 2*N, 1, 1).type(dtype)
        return prf

    # getting p_r positions from each p_c
    def _get_p_r(self, b, h, w, N, gb, dtype):
        rb = self.refer_band
        # (1, 2N, 1, 1)
        prf = self._gen_prf_grid(rb, gb, N, dtype)
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
        # x_pad: shape of (b, c, H, W)
        h_pad = x_pad.size(2)
        w_pad = x_pad.size(3)
        c = x_pad.size(1)
        # (b, c, h_pad*w_pad)
        # strech each spatial channel of x_pad as 1-D vector
        x_pad = x_pad.contiguous().view(b, c, -1)
        # (b, h, w, N)
        # transform spatial coord of p into the 1-D index
        index = p[..., :N]*w_pad + p[..., N:]
        # index_x = torch.clamp(p[..., :N], 0, h_pad-1)
        # index_y = torch.clamp(p[..., N:], 0, w_pad-1)
        # index = index_x * w_pad + index_y
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

# double PeakConv 2D Block definition
class DoublePeakConv2D(nn.Module):
    """ (2D PeakConv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, bias, rb, sig):
        super(DoublePeakConv2D, self).__init__()
        self.bias = bias
        self.refer_band = rb
        self.signal_type = sig
        # predifined guard bandwidth in range, Doppler and angle.
        gb_R = 2
        gb_D = 2
        gb_A = 2
        if self.signal_type == 'range_doppler':
            self.gb_h = gb_R
            self.gb_w = gb_D
        elif self.signal_type == 'angle_doppler':
            self.gb_h = gb_A
            self.gb_w = gb_D
        elif self.signal_type == 'range_angle':
            self.gb_h = gb_R
            self.gb_w = gb_A
        else:
            raise KeyError('signal type is not correct: {}'.format(self.signal_type))
        self.pk_conv1 = PKC2D_Sample(in_ch, out_ch, bias=self.bias,
                                   refer_band=self.refer_band,
                                   guard_band_h=self.gb_h,
                                   guard_band_w=self.gb_w,
                                   samping_type='v1')
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.pk_conv2 = PKC2D_Sample(out_ch, out_ch, bias=self.bias,
                                   refer_band=self.refer_band,
                                   guard_band_h=self.gb_h,
                                   guard_band_w=self.gb_w,
                                   samping_type='v1')
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
        '''
        self.double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                  pad=1, dil=1)
        '''
        self.double_pkc_block = DoublePeakConv2D(in_ch=128, out_ch=128,
                                                 bias=True,
                                                 rb=1,
                                                 sig=self.signal_type)
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

        # x2 = self.double_conv_block2(x1_down)
        x2 = self.double_pkc_block(x1_down)
        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x2_pad = F.pad(x2, (0, 1, 0, 0), "constant", 0)
            x2_down = self.doppler_max_pool(x2_pad)
        else:
            x2_down = self.max_pool(x2)

        x3 = self.single_conv_block1_1x1(x2_down)
        # return input of ASPP block + latent features
        return x2_down, x3


class PKCIn_AdaSample(nn.Module):
    """ 
    PKCIn-Net with different sampling strategies

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
