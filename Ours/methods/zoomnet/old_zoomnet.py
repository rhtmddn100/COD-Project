import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from utils.builder import MODELS
from utils.ops import cus_sample
from methods.module.pvtv2 import pvt_v2_b2


class Channel_Matching(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        self.c5_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
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


class GA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GA, self).__init__()
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


class GAB(nn.Module):
    def __init__(self, channel):
        super(GAB, self).__init__()
        self.weak_gra = GA(channel, channel)
        self.little_gra = GA(channel, 8)
        self.medium_gra = GA(channel, 4)
        self.strong_gra = GA(channel, 2)

    def forward(self, x, y):
        # reverse guided block
        y = torch.sigmoid(y)

        # three group-reversal attention blocks
        x, y = self.weak_gra(x, y)
        x, y = self.little_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)

        return y



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



class CCBR(nn.Module):
    def __init__(self):
        super(CCBR, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1), nn.ReLU(True),
        )
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
    
    def forward(self, x, y):
        x_cat = torch.cat((x, y), 1)
        x = x + self.conv1(x_cat)
        y = y + self.conv2(x)

        return y




@MODELS.register()
class ZoomNet(BasicModelClass):
    def __init__(self):
        super().__init__()

        ##Feature Encoder##
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        ##Sketch Decoder##
        self.CM = Channel_Matching(out_c=64)  # [c5, c4, c3, c2]
        self.s4 = [CAB(64, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]
        self.s3 = [CAB(64, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]
        self.s2 = [CAB(64, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]
        self.s1 = [CAB(64, 3, 4, bias=False, act=nn.PReLU()) for _ in range(2)]
        self.s4 = nn.Sequential(*self.s4)
        self.s3 = nn.Sequential(*self.s3)
        self.s2 = nn.Sequential(*self.s2)
        self.s1 = nn.Sequential(*self.s1)
        self.out_layer_00 = ConvBNReLU(64, 64, 3, 1, 1)
        self.out_layer_01 = nn.Conv2d(64, 1, 1)

        ##Refine Decoder##
        self.combi_layer_4 = ConvBNReLU(128, 64, 3, 1, 1)
        self.combi_layer_3 = ConvBNReLU(192, 64, 3, 1, 1)
        self.combi_layer_2 = ConvBNReLU(192, 64, 3, 1, 1)
        self.combi_layer_1 = ConvBNReLU(128, 64, 3, 1, 1)
        self.Refine4 = GAB(64)
        self.Refine3 = GAB(64)
        self.Refine2 = GAB(64)
        self.Refine1 = GAB(64)

        ##Retoouch Decoder##
        self.Retouch4 = CCBR()
        self.Retouch3 = CCBR()
        self.Retouch2 = CCBR()
        self.Retouch1 = CCBR()


        '''
        self.final_comb_5 = nn.Sequential(
		    ConvBNReLU(2, 1, kernel_size=1),
			ConvBNReLU(1, 1, kernel_size=1) )
        self.final_comb_4 = nn.Sequential(
		    ConvBNReLU(2, 1, kernel_size=1),
			ConvBNReLU(1, 1, kernel_size=1)	)
        self.final_comb_3 = nn.Sequential(
		    ConvBNReLU(2, 1, kernel_size=1),
			ConvBNReLU(1, 1, kernel_size=1))
        self.final_comb_2 = nn.Sequential(
		    ConvBNReLU(2, 1, kernel_size=1),
			ConvBNReLU(1, 1, kernel_size=1))
        '''

    # def encoder_translayer(self, x):
    #     pvt = self.backbone(x)
        
    #     trans_feats = self.CM(pvt)
    #     return trans_feats

    def body(self, s_scale):

        # Feature Encoder
            # shape => s_scale: [2,3,384,384], m_scale: [2,3,576,576], l_scale: [2,3,768,768]
            # s_trans_feats (tuple type) : 0:[2,64,12,12}, 1:[2,64,24,24], 2:[2,64,48,48], 3:[2,64,96,96]
            # m_trans_feats (tuple type) : 0:[2,64,18,18}, 1:[2,64,36,36], 2:[2,64,72,72], 3:[2,64,144,144]
        # s_trans_feats = self.encoder_translayer(s_scale) # x1.0
        pvt = self.backbone(s_scale)


        ## Sketch Decoder ##
        #channel matching
        trans_feats = self.CM(pvt)

        f4 = self.s4(trans_feats[0])  # [bs,64,12,12]
        f3 = cus_sample(f4, mode="scale", factors=2)  # [bs,64,24,24]
        f3 = self.s3(f3 + trans_feats[1])  # [bs,64,24,24]
        f2 = cus_sample(f3, mode="scale", factors=2)  # [bs,64,48,48]
        f2 = self.s2(f2 + trans_feats[2])  # [bs,64,48,48]
        f1 = cus_sample(f2, mode="scale", factors=2)  # [bs,64,96,96]
        f1 = self.s1(f1 + trans_feats[3])  # [bs,64,96,96]

        # coarse prediction map
        p5 = self.out_layer_01(self.out_layer_00(f1)) # [bs,1,96,96]
        Coarse_pred = F.interpolate(p5, scale_factor=4, mode='bilinear') # [bs,1,384,384]


        ## Refine Decoder ##
        # ---- level 4 ----
        guidance_5 = F.interpolate(p5, scale_factor=0.25, mode='bilinear') # [bs,1,24,24]
        g4 = torch.cat((cus_sample(f4, mode="scale", factors=2), f3), 1) # [bs,128,24,24]
        g4 = self.combi_layer_4(g4) # [bs,64,24,24]
        ra4_feat = self.Refine4(g4, guidance_5)
        p4 = ra4_feat + guidance_5
        p4_pred = F.interpolate(p4, scale_factor=16, mode='bilinear')  # Sup-2 (bs, 1, 24, 24) -> (bs, 1, 384, 384)

        # ---- level 3 ----
        guidance_4 = F.interpolate(p4, scale_factor=2, mode='bilinear')
        g3 = torch.cat((cus_sample(f4, mode="scale", factors=4), cus_sample(f3, mode="scale", factors=2), f2), 1) # [bs,192,48,48]
        g3 = self.combi_layer_3(g3) # [bs,64,48,48]
        ra3_feat = self.Refine3(g3, guidance_4)
        p3 = ra3_feat + guidance_4
        p3_pred = F.interpolate(p3, scale_factor=8, mode='bilinear')  # Sup-3 (bs, 1, 48, 48) -> (bs, 1, 384, 384)

        # ---- level 2 ----
        guidance_3 = F.interpolate(p3, scale_factor=2, mode='bilinear')
        g2 = torch.cat((cus_sample(f3, mode="scale", factors=4), cus_sample(f2, mode="scale", factors=2), f1), 1) # [bs,192,96,96]
        g2 = self.combi_layer_2(g2) # [bs,64,96,96]
        ra2_feat = self.Refine2(g2, guidance_3)
        p2 = ra2_feat + guidance_3
        p2_pred = F.interpolate(p2, scale_factor=4, mode='bilinear')   # Sup-4 (bs, 1, 96, 96) -> (bs, 1, 384, 384)

        # ---- level 1 ----
        guidance_2 = F.interpolate(p2, scale_factor=2, mode='bilinear')
        g1 = torch.cat((cus_sample(f2, mode="scale", factors=4), cus_sample(f1, mode="scale", factors=2)), 1) # [bs,1,192,192]
        g1 = self.combi_layer_1(g1) # [bs,64,192,192]
        ra1_feat = self.Refine1(g1, guidance_2)
        p1 = ra1_feat + guidance_2
        p1_pred = F.interpolate(p1, scale_factor=2, mode='bilinear')  # Sup-5 (bs, 1, 192, 192) -> (bs, 1, 384, 384)


        ## Retouch Decoder ##
        p1_map = F.interpolate(p1, scale_factor=0.125, mode='bilinear') 
        M_4 = self.Retouch4(p1_map, p4)
        M_4_pred = F.interpolate(M_4, scale_factor=16, mode='bilinear')
        
        M_4 = F.interpolate(M_4, scale_factor=2, mode='bilinear')
        M_3 = self.Retouch3(M_4, p3)
        M_3_pred = F.interpolate(M_3, scale_factor=8, mode='bilinear')

        M_3 = F.interpolate(M_3, scale_factor=2, mode='bilinear')
        M_2 = self.Retouch2(M_3, p2)
        M_2_pred = F.interpolate(M_2, scale_factor=4, mode='bilinear')
	
        M_2 = F.interpolate(M_2, scale_factor=2, mode='bilinear')
        M_1 = self.Retouch1(M_2, p1)
        M_1_pred = F.interpolate(M_1, scale_factor=2, mode='bilinear')


        return dict(p5=Coarse_pred, p4=p4_pred, p3=p3_pred, p2=p2_pred, p1=p1_pred, M_4=M_4_pred, M_3=M_3_pred, M_2=M_2_pred, M_1=M_1_pred)



    def train_forward(self, data, **kwargs):
        assert not {"image1.0", "mask"}.difference(set(data)), set(data)

        output = self.body(
            s_scale=data["image1.0"],
        )
        loss, loss_str = self.cal_loss(
            all_preds=output,
            gts=data["mask"],
            iter_percentage=kwargs["curr"]["iter_percentage"],
        )
        return dict(sal=output["M_1"].sigmoid()), loss, loss_str
    
    def test_forward(self, data, **kwargs):
        output = self.body(
            s_scale=data["image1.0"],
        )
        return output["M_1"]

    def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
        # ual_coef = get_coef(iter_percentage, method)

        losses = []
        loss_str = []
        # for main
        #loss function: weighted BCE for each layer + ual loss for the last layer
        for name, preds in all_preds.items():
            resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])

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