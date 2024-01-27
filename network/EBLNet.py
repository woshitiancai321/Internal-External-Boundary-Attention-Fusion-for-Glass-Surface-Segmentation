import torch
import torch.nn as nn
import torch.nn.functional as F

from network import resnet_d as Resnet_Deep
from network.resnext import resnext101_32x8
from network.nn.mynn import Norm2d
from network.nn.contour_point_gcn import ContourPointGCN
from network.nn.operators import _AtrousSpatialPyramidPoolingModule
from network.Fullyattention import FullyAttentionalBlock

class body_han1(nn.Module):#hands-change
    def __init__(self, inplane, skip_num, norm_layer):
        super(body_han1, self).__init__()
        self.skip_mum = skip_num
        plane=256
        self.pre_extractor1 = nn.Sequential(
              nn.Conv2d(512, 256, kernel_size=3,padding=1,
                         groups=1, bias=False),
              norm_layer(256),
              nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Linear(plane, plane)
        self.conv2 = nn.Linear(plane, plane)
        self.conv = nn.Sequential(nn.Conv2d(plane, plane, 3, stride=1, padding=1, bias=False),
                                  norm_layer(plane),
                                  nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, body, ex_layer):  # 200        # 100
        batch_size, _, height, width = body.size()

        feat_h = body.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)
        feat_w = body.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)
        encode_h = self.conv1(F.avg_pool2d(ex_layer, [1, width]).view(batch_size, -1, height).permute(0, 2, 1).contiguous())
        encode_w = self.conv2(F.avg_pool2d(ex_layer, [height, 1]).view(batch_size, -1, width).permute(0, 2, 1).contiguous())

        energy_h = torch.matmul(feat_h, encode_h.repeat(width, 1, 1))
        energy_w = torch.matmul(feat_w, encode_w.repeat(height, 1, 1))
        full_relation_h = self.softmax(energy_h)  # [b*w, c, c]
        full_relation_w = self.softmax(energy_w)

        full_aug_h = torch.bmm(full_relation_h, feat_h).view(batch_size, width, -1, height).permute(0, 2, 3, 1)
        full_aug_w = torch.bmm(full_relation_w, feat_w).view(batch_size, height, -1, width).permute(0, 2, 1, 3)
        out = self.gamma * (full_aug_h + full_aug_w) + body
        out = self.conv(out)
        return out
class Edge_extractorWofirstext1(nn.Module):#hands-change
    def __init__(self, inplane, skip_num, norm_layer):
        super(Edge_extractorWofirstext1, self).__init__()
        self.skip_mum = skip_num
        self.extractor1 = nn.Sequential(
            nn.Conv2d(384, inplane, kernel_size=3,
                      padding=1, groups=8, bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )
        self.extractor2 = nn.Sequential(
            nn.Conv2d(384, inplane, kernel_size=3,
                      padding=1, groups=8, bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )
        self.extractor3 = nn.Sequential(
            nn.Conv2d(384, inplane, kernel_size=3,
                      padding=1, groups=8, bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )
        self.extractor4 = nn.Sequential(
            nn.Conv2d(384, inplane, kernel_size=3,
                      padding=1, groups=8, bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )
        self.extractor5 = nn.Sequential(
            nn.Conv2d(384, inplane, kernel_size=3,
                      padding=1, groups=8, bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )
        self.extractor6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=False),
                                  norm_layer(128),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
                                  norm_layer(128),
                                  nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=False),
                                  norm_layer(128),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
                                  norm_layer(128),
                                  nn.ReLU())
        #self.pre_extractor3 =  nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.fully1 = FullyAttentionalBlock(128)
        self.fully2 = FullyAttentionalBlock(128)
        self.mlp_internal = nn.Sequential(nn.Conv2d(256, 32, 1, 1, 0),
                                       nn.BatchNorm2d(32), nn.ReLU(),
                                       nn.Conv2d(32, 1, 1, 1, 0))
        self.gap_internal = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp_external = nn.Sequential(nn.Conv2d(256, 32, 1, 1, 0),
                                       nn.BatchNorm2d(32), nn.ReLU(),
                                       nn.Conv2d(32, 1, 1, 1, 0))
        self.gap_external = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp_external1 = nn.Sequential(nn.Conv2d(256, 256, 3, 2, 0),
                                       nn.BatchNorm2d(256), nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, 2, 0))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, aspp, layer1):#,parameter):  # 200        # 100
        seg_edge = torch.cat([F.interpolate(aspp, size=layer1.size()[2:], mode='bilinear',
                                            align_corners=True), layer1], dim=1)  # 200
        
        
        seg_edge1=self.extractor5(seg_edge)
        #seg_subedge=seg_final_fullyexter-seg_final_fullyinter
        
        ppp=torch.cat([seg_edge1,layer1],dim=1)
        ppp1 = self.extractor1(ppp)
        seg_final_fullyinter1=self.fully1(self.conv1(ppp1))
        seg_final_fullyinter=self.extractor3(torch.cat([ppp1,seg_final_fullyinter1],dim=1))
        # seg_final_fullyinter=self.fully1(seg_final_fullyinter)
        ppp2 = self.extractor2(ppp)
        seg_final_fullyexter1=self.fully2(self.conv2(ppp2))
        seg_final_fullyexter=self.extractor4(torch.cat([ppp2,seg_final_fullyexter1],dim=1))
        # concatenation=self.fusion3(torch.cat((seg_final_fullyinter,seg_final_fullyexter),1))
        #weight_inter = self.mlp_internal(self.gap_internal(aspp))
        aspp1=self.mlp_external1(aspp)
        weight_exter = self.mlp_external(self.gap_external(aspp1))
        #weight_inter=weight_inter.view(B,-1,H*W)

        softmax_weight = F.sigmoid(weight_exter)
        energy_internal =  seg_final_fullyinter*(1-softmax_weight).expand_as(seg_final_fullyinter)
        energy_external=seg_final_fullyexter * softmax_weight.expand_as(seg_final_fullyexter)
        #kkkkk=torch.cat([energy_internal,energy_external],dim=1)
        #kkkkk = self.extractor6(kkkkk)#+self.gamma*seg_edge1
        seg_edge=energy_internal+energy_external
        seg_edge = self.extractor6(seg_edge)
        seg_body = F.interpolate(aspp, layer1.size()[2:], mode='bilinear', align_corners=True) - seg_final_fullyinter
        return seg_edge1,seg_edge, seg_body,seg_final_fullyinter,seg_final_fullyexter


class EBLNet(nn.Module):
    """
    Implement deeplabv3 plus module without depthwise conv
    A: stride=8
    B: stride=16
    with skip connection
    """

    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant='D',
                 skip='m1', skip_num=48, num_cascade=4,
                 num_points=96, threshold=0.8):
        super(EBLNet, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_mum = skip_num
        self.num_cascade = num_cascade
        self.num_points = num_points
        self.threshold = threshold

        if trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-101-32x8':
            resnet = resnext101_32x8()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError('Only support resnet50 and resnet101 for now')

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            print('Not using dilation')

        self.aspp = _AtrousSpatialPyramidPoolingModule(in_dim=2048, reduction_dim=256,
                                                       output_stride=8 if self.variant == 'D' else 16)

        if self.skip == 'm1':
            self.bot_fine = nn.Conv2d(256, 128, kernel_size=1, bias=False)#self.skip_mum
        elif self.skip == 'm2':
            self.bot_fine = nn.Conv2d(512, 128, kernel_size=1, bias=False)#self.skip_mum
        else:
            raise ValueError('Not a valid skip')
        self.body_fines = nn.ModuleList()
        for i in range(self.num_cascade):
            inchannels = 2 ** (11 - i)
            self.body_fines.append(nn.Conv2d(inchannels, 48, kernel_size=1, bias=False))
        self.body_fuse = [nn.Conv2d(256 + 48, 256, kernel_size=1, bias=False) for _ in range(self.num_cascade)]
        self.body_fuse = nn.ModuleList(self.body_fuse)

        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.edge_extractors = [Edge_extractorWofirstext1(256, norm_layer=Norm2d, skip_num=48)
                                for _ in range(self.num_cascade)]
        self.edge_extractors = nn.ModuleList(self.edge_extractors)

        #self.refines = [ContourPointGCN(256, self.num_points, self.threshold) for _ in range(self.num_cascade)]
        #self.refines = nn.ModuleList(self.refines)

        self.edge_out_pre = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.edge_out_pre = nn.ModuleList(self.edge_out_pre)
        self.edge_out1 = nn.ModuleList([nn.Conv2d(256, 1, kernel_size=1, bias=False)
                                       for _ in range(self.num_cascade)])
        self.edge_out_pre1 = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.edge_out_pre1 = nn.ModuleList(self.edge_out_pre)            

        self.body_out_pre = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.body_out_pre = nn.ModuleList(self.body_out_pre)
        self.body_out = nn.ModuleList([nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
                                       for _ in range(self.num_cascade)])

        self.final_seg_out_pre = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade - 1)]
        self.final_seg_out_pre.append(nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)))
        self.final_seg_out_pre = nn.ModuleList(self.final_seg_out_pre)
        self.final_seg_out = nn.ModuleList([nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
                                            for _ in range(self.num_cascade)])
        #han 
        self.fully_in_pre = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.fully_in_pre = nn.ModuleList(self.fully_in_pre)
        self.fully_out_pre= [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.fully_out_pre = nn.ModuleList(self.fully_out_pre)
        self.fully_seg_out1 = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.fully_seg_out2 = nn.Conv2d(256, 1, kernel_size=1, bias=False)

        self.body_han = [body_han1(256, norm_layer=Norm2d, skip_num=48)for _ in range(self.num_cascade)]
        self.body_han = nn.ModuleList(self.body_han)

    def forward(self, x, gts=None):

        x_size = x.size()  # 800
        feats = []
        feats.append(self.layer0(x))  # 200
        feats.append(self.layer1(feats[0]))  # 200, 200
        feats.append(self.layer2(feats[1]))  # 200, 200, 100
        feats.append(self.layer3(feats[2]))  # 200, 200, 100, 100
        feats.append(self.layer4(feats[3]))  # 200, 200, 100, 100, 100
        aspp = self.aspp(feats[-1])  # 100
        fine_size = feats[1].size()  # 200

        seg_edges = []
        seg_edge_outs = []
        seg_bodys = []
        seg_body_outs = []
        seg_finals = []
        seg_final_outs = []
        seg_final_fullyexters= []
        seg_final_fullyinters= []
        aspp = self.bot_aspp(aspp)  # 100
        final_fuse_feat = F.interpolate(aspp, size=fine_size[2:], mode='bilinear', align_corners=True)  # 200

        low_feat = self.bot_fine(feats[1])  # 200

        for i in range(self.num_cascade):
            if i == 0:
                last_seg_feat = aspp  # 100
            else:
                last_seg_feat = seg_finals[-1]  # 200
                last_seg_feat = F.interpolate(last_seg_feat, size=aspp.size()[2:],
                                              mode='bilinear', align_corners=True)  # 100

            seg_edge1,seg_edge, seg_body,seg_final_fullyinter,seg_final_fullyexter = self.edge_extractors[i](last_seg_feat, low_feat)  # 200

            high_fine = F.interpolate(self.body_fines[i](feats[-(i + 1)]), size=fine_size[2:], mode='bilinear',align_corners=True)  # 200
            seg_body = self.body_fuse[i](torch.cat([seg_body, high_fine], dim=1))  # 200

            seg_body_pre = self.body_out_pre[i](seg_body)#seg_body_hab
            seg_body_out = F.interpolate(self.body_out[i](seg_body_pre), size=x_size[2:],
                                         mode='bilinear', align_corners=True)  # 800
            seg_bodys.append(seg_body_pre)
            seg_body_outs.append(seg_body_out)

            seg_edge_pre1 = self.edge_out_pre1[i](seg_edge1)  # 200
            seg_edge_out_pre1 = self.edge_out1[i](seg_edge_pre1)
            seg_edge_out = F.interpolate(seg_edge_out_pre1, size=x_size[2:],
                                           mode='bilinear', align_corners=True)  # 800
            seg_edges.append(seg_edge_pre1)
            seg_edge_outs.append(seg_edge_out)
            seg_out = seg_body + seg_final_fullyinter  # 200#seg_bode_han
            #seg_out=self.body_han[i](seg_out,seg_edge)#hands

            seg_final_fullyinter =  self.fully_in_pre[i](seg_final_fullyinter) 
            seg_final_fullyexter =self.fully_out_pre[i](seg_final_fullyexter)   
            xxxx1=self.fully_seg_out1(seg_final_fullyinter) 
            xxxx2=self.fully_seg_out2(seg_final_fullyexter)
            seg_out=self.body_han[i](seg_out,seg_edge)#hands

            if i >= self.num_cascade - 1:
                seg_final_pre = self.final_seg_out_pre[i](torch.cat([final_fuse_feat, seg_out], dim=1))
            else:
                seg_final_pre = self.final_seg_out_pre[i](seg_out)
            seg_final_out = F.interpolate(self.final_seg_out[i](seg_final_pre), size=x_size[2:],
                                          mode='bilinear', align_corners=True)
            seg_finals.append(seg_final_pre)
            seg_final_outs.append(seg_final_out)
            #hands
            seg_final_fullyinter= F.interpolate(xxxx1, size=x_size[2:],
                                          mode='bilinear', align_corners=True)
            seg_final_fullyexter= F.interpolate(xxxx2, size=x_size[2:],
                                          mode='bilinear', align_corners=True)
            seg_final_fullyinters.append(seg_final_fullyinter)
            seg_final_fullyexters.append(seg_final_fullyexter)
        if self.training:
            return self.criterion((seg_final_outs, seg_body_outs, seg_edge_outs,seg_final_fullyinters,seg_final_fullyexters), gts)
        return seg_final_outs[-1]


def EBLNet_resnet50_os8(num_classes, criterion, num_cascade=1, num_points=96, threshold=0.85):
    """
    ResNet-50 Based Network with stride=8 and cascade model
    """
    return EBLNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1',
                  num_cascade=num_cascade, num_points=num_points, threshold=threshold)


def EBLNet_resnet101_os8(num_classes, criterion, num_cascade=1, num_points=96, threshold=0.85):
    """
    ResNet-101 Based Network with stride=8 and cascade model
    """
    return EBLNet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', skip='m1',
                  num_cascade=num_cascade, num_points=num_points, threshold=threshold)


def EBLNet_resnet50_os16(num_classes, criterion, num_cascade=1, num_points=96, threshold=0.85):
    """
    ResNet-50 Based Network with stride=16 and cascade model
    """
    return EBLNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D16', skip='m1',
                  num_cascade=num_cascade, num_points=num_points, threshold=threshold)


def EBLNet_resnext101_os8(num_classes, criterion, num_cascade=1, num_points=96, threshold=0.85):
    """
    ResNeXt-101 Based Network with stride=8 and cascade model
    """
    return EBLNet(num_classes, trunk='resnext-101-32x8', criterion=criterion, variant='D', skip='m1',
                  num_cascade=num_cascade, num_points=num_points, threshold=threshold)

