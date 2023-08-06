import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from utils.builder import MODELS
from utils.ops import cus_sample
from methods.module.pvtv2 import pvt_v2_b2


class ASPP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ASPP, self).__init__()
        self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
        self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
        # self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.fuse = ConvBNReLU(4 * out_dim, out_dim, 3, 1, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        # conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
        # return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4), 1))


class TransLayer(nn.Module):
    def __init__(self, out_c, last_module=ASPP):
        super().__init__()
        self.c5_down = nn.Sequential(
            # ConvBNReLU(2048, 256, 3, 1, 1),
            last_module(in_dim=512, out_dim=out_c),
        )
        self.c4_down = nn.Sequential(ConvBNReLU(320, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(128, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))
        # self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

    def forward(self, xs):
        # assert isinstance(xs, (tuple, list))
        assert len(xs) == 4
        c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        # c1 = self.c1_down(c1)
        # return c5, c4, c3, c2, c1
        return c5, c4, c3, c2

class SIU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_m_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(                                                                        ###
            ConvBNReLU(3 * in_dim, in_dim, 1),                                                             ###
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),                                                           ###
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),                                                           ###
            nn.Conv2d(in_dim, 3, 1),                                                                       ###
        )                                                                                                  ###

    def forward(self, l, m, s, return_feats=False):
        # x 1.0
        # s = self.conv_m(s)
        s = self.conv_s(s)

        tgt_size = s.shape[2:]
        # x 1.5
        m = self.conv_m_pre_down(m)
        m = F.adaptive_max_pool2d(m, tgt_size) + F.adaptive_avg_pool2d(m, tgt_size)
        m = self.conv_m_post_down(m)
        # x 2.0
        l = self.conv_l_pre_down(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv_l_post_down(l)

        attn = self.trans(torch.cat([s, m, l], dim=1))                                                     ###
        attn_s, attn_m, attn_l = torch.softmax(attn, dim=1).chunk(3, dim=1)                                ###
        lms = attn_s * s + attn_m * m + attn_l * l                                                         ###

        if return_feats:
            return lms, dict(attn_s=attn_s, attn_m=attn_m, attn_l=attn_l, s=s, m=m, l=l)
        return lms

class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel//subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        else:
            raise Exception("Invalid Channel")

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y

class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.little_gra = GRA(channel, 8)
        self.medium_gra = GRA(channel, 4)
        self.strong_gra = GRA(channel, 2)

    def forward(self, x, y):
        # reverse guided block
        y = -1 * (torch.sigmoid(y)) + 1

        # three group-reversal attention blocks
        x, y = self.weak_gra(x, y)
        x, y = self.little_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)

        return y

class HMU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []

        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(3, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(3, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(2, dim=1))

        out = torch.cat([o[0] for o in outs], dim=1)
        gate = self.gate_genator(torch.cat([o[-1] for o in outs], dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias))
        modules_body.append(act)
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


@MODELS.register()
class ZoomNet(BasicModelClass):
    def __init__(self):
        super().__init__()

        # load pvt
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.translayer = TransLayer(out_c=64)  # [c5, c4, c3, c2, c1]
        # self.merge_layers = nn.ModuleList([SIU(in_dim=in_c) for in_c in (64, 64, 64, 64, 64)])
        self.merge_layers = nn.ModuleList([SIU(in_dim=in_c) for in_c in (64, 64, 64, 64, 64)])

        self.d5 = [CAB(64, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]
        self.d4 = [CAB(64, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]
        self.d3 = [CAB(64, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]
        self.d2 = [CAB(64, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]
        self.d5 = nn.Sequential(*self.d5)
        self.d4 = nn.Sequential(*self.d4)
        self.d3 = nn.Sequential(*self.d3)
        self.d2 = nn.Sequential(*self.d2)
        self.out_layer_00 = ConvBNReLU(64, 64, 3, 1, 1)
        self.out_layer_01 = nn.Conv2d(64, 1, 1)

        self.RS5 = ReverseStage(64)
        self.RS4 = ReverseStage(64)
        self.RS3 = ReverseStage(64)
        self.RS2 = ReverseStage(64)
        # self.RS1 = ReverseStage(64)
        self.RS1 = GRA(1, 1)

    def encoder_translayer(self, x):
        pvt = self.backbone(x)
        # x1 = pvt[0]
        # x2 = pvt[1]
        # x3 = pvt[2]
        # x4 = pvt[3]
        # en_feats = self.shared_encoder(x)
        trans_feats = self.translayer(pvt)
        return trans_feats

    def body(self, s_scale, m_scale, l_scale):
        # shape => s_scale: [2,3,384,384], m_scale: [2,3,576,576], l_scale: [2,3,768,768]
        # s_trans_feats (tuple type) : 0:[2,64,12,12}, 1:[2,64,24,24], 2:[2,64,48,48], 3:[2,64,96,96]
        # m_trans_feats (tuple type) : 0:[2,64,18,18}, 1:[2,64,36,36], 2:[2,64,72,72], 3:[2,64,144,144]
        s_trans_feats = self.encoder_translayer(s_scale) # x1.0
        m_trans_feats = self.encoder_translayer(m_scale) # x1.5
        l_trans_feats = self.encoder_translayer(l_scale) # x2.0

        feats = []
        for s, m, l, layer in zip(s_trans_feats, m_trans_feats, l_trans_feats, self.merge_layers):
            siu_outs = layer(s=s, m=m, l=l)
            feats.append(siu_outs)
        
        #feats[0:2] = outputs of SIU 3~5

        x1 = self.d5(feats[0])  # [bs,64,12,12]
        x2 = cus_sample(x1, mode="scale", factors=2)  # [bs,64,24,24]
        x2 = self.d4(x2 + feats[1])  # [bs,64,24,24]
        x3 = cus_sample(x2, mode="scale", factors=2)  # [bs,64,48,48]
        x3 = self.d3(x3 + feats[2])  # [bs,64,48,48]
        x4 = cus_sample(x3, mode="scale", factors=2)  # [bs,64,96,96]
        x4 = self.d2(x4 + feats[3])  # [bs,64,96,96]
        # x = cus_sample(x, mode="scale", factors=2)
        # # x = self.d1(x + feats[4])
        # # x = cus_sample(x, mode="scale", factors=2)

        #x = output of HMU 3
        #coarse map
        S_g = self.out_layer_01(self.out_layer_00(x3)) # [bs,1,96,96]
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear') # [bs,1,384,384]

        # ---- reverse stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear') # [bs,1,12,12]
        ra4_feat = self.RS5(x1, guidance_g)
        S_5 = ra4_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra3_feat = self.RS4(x2, guidance_5)
        S_4 = ra3_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse stage 3 ----
       # guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        guidance_4 = S_g
        ra2_feat = self.RS3(x3, guidance_4)
        S_3 = ra2_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse stage 2 ----
        guidance_3 = F.interpolate(S_3, scale_factor=2, mode='bilinear')
        ra1_feat = self.RS2(x4, guidance_3)
        S_2 = ra1_feat + guidance_3
        S_2_pred = F.interpolate(S_2, scale_factor=4, mode='bilinear')  # Sup-5 (bs, 1, 88, 88) -> (bs, 1, 352, 352)

        # # ---- decoder type d ---- #seperate decoding of high&low level feaures
        high_map = F.interpolate(S_4, scale_factor=4, mode='bilinear')
        low_map = S_2
        _, final_map = self.RS1(low_map, high_map)
        S_1_pred = F.interpolate(final_map, scale_factor=4, mode='bilinear')


        # # ---- reverse stage 1 ----
        # guidance_2 = F.interpolate(S_2, scale_factor=2, mode='bilinear')
        # ra0_feat = self.RS1(feats[4], guidance_2)
        # S_1 = ra0_feat + guidance_2
        # S_1_pred = F.interpolate(S_1, scale_factor=2, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # return dict(seg=logits)
     #   return dict(S_g=S_g_pred, S_5=S_5_pred, S_4=S_4_pred, S_3=S_3_pred, S_2=S_2_pred)

        return dict(S_g=S_g_pred, S_5=S_5_pred, S_4=S_4_pred, S_3=S_3_pred, S_2=S_2_pred, S_1=S_1_pred)

    def train_forward(self, data, **kwargs):
        assert not {"image1.0", "image1.5", "image2.0", "mask"}.difference(set(data)), set(data)

        output = self.body(
            s_scale=data["image1.0"],
            m_scale=data["image1.5"],
            l_scale=data["image2.0"],
        )
        loss, loss_str = self.cal_loss(
            all_preds=output,
            gts=data["mask"],
            iter_percentage=kwargs["curr"]["iter_percentage"],
        )
        return dict(sal=output["S_2"].sigmoid()), loss, loss_str
    
    def test_forward(self, data, **kwargs):
        output = self.body(
            s_scale=data["image1.0"],
            m_scale=data["image1.5"],
            l_scale=data["image2.0"],
        )
        return output["S_2"]

    def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
        ual_coef = get_coef(iter_percentage, method)

        losses = []
        loss_str = []
        # for main
        #loss function: weighted BCE for each layer + ual loss for the last layer
        for name, preds in all_preds.items():
            resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])

            ## FREEZE AREA FOR EXPERIMENT
            # sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="none")
            # weit = 1 + 5 * torch.abs(F.avg_pool2d(resized_gts, kernel_size=31, stride=1, padding=15) - resized_gts)
            # # wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
            # w_sod_loss = (weit * sod_loss).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
            # w_sod_loss = w_sod_loss
            # # losses.append(w_sod_loss)

            # preds = torch.sigmoid(preds)
            # inter = ((preds * resized_gts) * weit).sum(dim=(2, 3))
            # union = ((preds + resized_gts) * weit).sum(dim=(2, 3))
            # wiou = 1 - (inter + 1) / (union - inter + 1)
            
            # total_loss = (w_sod_loss + wiou).mean()
            # losses.append(total_loss)

            # loss_str.append(f"{name}_wBCE+wIOU: {total_loss.item():.5f}")
            ## TILL HERE

            sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="none")
            weit = 1 + 5 * torch.abs(F.avg_pool2d(resized_gts, kernel_size=31, stride=1, padding=15) - resized_gts)
            # wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
            w_sod_loss = (weit * sod_loss).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
            w_sod_loss = w_sod_loss
            # losses.append(w_sod_loss)

            preds = torch.sigmoid(preds)
            inter = ((preds * resized_gts) * weit).sum(dim=(2, 3))
            union = ((preds + resized_gts) * weit).sum(dim=(2, 3))
            wiou = 1 - (inter + 1) / (union - inter + 1)
            
            total_loss = (w_sod_loss + wiou).mean()
            losses.append(total_loss)

            loss_str.append(f"{name}_wBCE+wIOU: {total_loss.item():.5f}")

            # if name == 'S_2':
            #     ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
            #     ual_loss *= ual_coef
            #     losses.append(ual_loss)
            #     loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")        
        return sum(losses), " ".join(loss_str)

    def get_grouped_params(self):
        param_groups = {}
        param_groups.setdefault("pretrained", [])
        param_groups.setdefault("fixed", [])
        for name, param in self.named_parameters():
            if name.startswith("backbone"):
                param_groups.setdefault("pretrained", []).append(param)
            # elif name.startswith("backbone"):
            #     param_groups.setdefault("fixed", []).append(param)
            else:
                param_groups.setdefault("retrained", []).append(param)
        return param_groups


@MODELS.register()
class ZoomNet_CK(ZoomNet):
    def __init__(self):
        super().__init__()
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def encoder(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x0, x1, x2, x3, x4 = self.shared_encoder(x)
        return x0, x1, x2, x3, x4

    def trans(self, x0, x1, x2, x3, x4):
        x5, x4, x3, x2, x1 = self.translayer([x0, x1, x2, x3, x4])
        return x5, x4, x3, x2, x1

    def decoder(self, x5, x4, x3, x2, x1):
        x = self.d5(x5)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d4(x + x4)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d3(x + x3)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d2(x + x2)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d1(x + x1)
        x = cus_sample(x, mode="scale", factors=2)
        logits = self.out_layer_01(self.out_layer_00(x))
        return logits

    def body(self, l_scale, m_scale, s_scale):
        l_trans_feats = checkpoint(self.encoder, l_scale, self.dummy)
        m_trans_feats = checkpoint(self.encoder, m_scale, self.dummy)
        s_trans_feats = checkpoint(self.encoder, s_scale, self.dummy)
        l_trans_feats = checkpoint(self.trans, *l_trans_feats)
        m_trans_feats = checkpoint(self.trans, *m_trans_feats)
        s_trans_feats = checkpoint(self.trans, *s_trans_feats)

        feats = []
        for layer_idx, (l, m, s) in enumerate(zip(l_trans_feats, m_trans_feats, s_trans_feats)):
            siu_outs = checkpoint(self.merge_layers[layer_idx], l, m, s)
            feats.append(siu_outs)

        logits = checkpoint(self.decoder, *feats)
        return dict(seg=logits)
