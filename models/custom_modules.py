import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class CBAM(nn.Module):
    def __init__(self, c1, reduction=16):
        super(CBAM, self).__init__()
        self.c1 = c1
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(c1, c1 // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(c1 // reduction, c1, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()
        
        # 空间注意力
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_out = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_out
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid_spatial(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_out
        
        return x 

class ResCBAM(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ResCBAM, self).__init__()
        self.c1 = c1
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(c1, c1 // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(c1 // reduction, c1, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()
        
        # 空间注意力
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
        
        # 可学习的残差连接权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        identity = x
        
        # 通道注意力
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_out = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_out
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid_spatial(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_out
        
        # 使用可学习的残差连接
        return x * self.alpha + identity * (1 - self.alpha)

class MinResCBAM(nn.Module):
    def __init__(self, c1, reduction=16):
        super(MinResCBAM, self).__init__()
        self.c1 = c1

        # 通道注意力：平均池化 + 最大池化 + 最小池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.min_pool = lambda x: -torch.max(-x, dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]  # 最小池化实现
        
        # 多统计量融合权重（可学习参数）
        self.channel_weight = nn.Parameter(torch.ones(3) / 3)  # 初始权重平均分配
        
        self.fc1 = nn.Conv2d(c1, c1 // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(c1 // reduction, c1, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # 空间注意力：均值 + 最大值 + 最小值
        self.conv = nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

        # 动态残差权重生成器
        self.alpha_gen = nn.Sequential(
            nn.Conv2d(c1, c1//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1//4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        # ========== 通道注意力 ==========
        # 获取三种池化结果
        avg_val = self.avg_pool(x)
        max_val = self.max_pool(x)
        min_val = self.min_pool(x)  # 最小池化
        
        # 可学习的统计量融合
        combined = torch.stack([avg_val, max_val, min_val], dim=0)
        weights = torch.softmax(self.channel_weight, dim=0)  # 归一化权重
        fused_stat = (combined * weights.view(3,1,1,1,1)).sum(dim=0)
        
        # 生成通道注意力
        channel_att = self.sigmoid_channel(self.fc2(self.relu(self.fc1(fused_stat))))
        x = x * channel_att

        # ========== 空间注意力 ==========
        # 沿通道维度计算均值、最大值、最小值
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        min_map, _ = torch.min(x, dim=1, keepdim=True)  # 新增最小值统计
        
        spatial_feat = torch.cat([avg_map, max_map, min_map], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv(spatial_feat))
        x = x * spatial_att

        # ========== 动态残差连接 ==========
        alpha = self.alpha_gen(x)  # 根据注意力结果生成权重
        return x * alpha + identity * (1 - alpha)

class IMinCBAM(nn.Module):
    def __init__(self, c1, reduction=16):
        super().__init__()
        # 原CBAM分支
        self.original_cbam = ResCBAM(c1, reduction)
        
        # 最小池化分支（独立参数）
        self.min_channel_att = nn.Sequential(
            nn.AdaptiveMinPool2d(1),  # 最小池化
            nn.Conv2d(c1, c1//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(c1//reduction, c1, 1),
            nn.Sigmoid()
        )
        self.min_spatial_att = nn.Sequential(
            nn.Conv2d(1, 1, 7, padding=3),  # 独立空间卷积
            nn.Sigmoid()
        )
        
        # 自适应融合参数
        self.alpha = nn.Parameter(torch.tensor([0.5]))  # 可学习融合系数

    def forward(self, x):
        original_out = self.original_cbam(x)
        min_channel = self.min_channel_att(x)
        channel_out = x * min_channel
        
        min_spatial, _ = torch.min(channel_out, dim=1, keepdim=True)
        spatial_out = self.min_spatial_att(min_spatial)
        min_out = channel_out * spatial_out
        
        return original_out * self.alpha + min_out * (1 - self.alpha)
        

class ProtoNet(nn.Module):
    def __init__(self, c1, c2=None, prototypes=2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2 or c1
        self.prototypes = prototypes
        
        # 初始化原型向量
        self.proto_features = nn.Parameter(F.normalize(torch.randn(prototypes, c1), dim=1))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 简化编码器结构，减少dropout使用
        self.encoder = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            # 仅保留一个dropout
            nn.Dropout(0.1)
        )
        
        # 简化投影器
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 可学习的残差权重
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        identity = x
        batch_size, c, h, w = x.shape
        
        # 应用编码器
        feat = self.encoder(x)
        
        # 计算原型相似度
        flat_feats = feat.permute(0, 2, 3, 1).reshape(-1, c)
        flat_feats = F.normalize(flat_feats, dim=1)
        proto_sim = torch.matmul(flat_feats, F.normalize(self.proto_features, dim=1).t()) / self.temperature
        proto_attention = F.softmax(proto_sim, dim=1).reshape(batch_size, h, w, self.prototypes)
        proto_attention = proto_attention.permute(0, 3, 1, 2)
        
        # 分离背景和前景注意力
        background_attn = proto_attention[:, 0:1]
        foreground_attn = proto_attention[:, 1:2] if self.prototypes == 2 else torch.max(proto_attention[:, 1:], dim=1, keepdim=True)[0]
        
        # 应用注意力增强
        enhanced_features = x * (1 + foreground_attn * self.a) - x * (background_attn * self.b)
        
        # 使用可学习的残差连接
        return enhanced_features 
    
    def __deepcopy__(self, memo):
        # 创建一个新的模块实例
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # 复制所有状态
        for k, v in self.__dict__.items():
            # 跳过存储的临时张量
            if k == 'contrast_logits':
                setattr(result, k, None)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        
        return result
    
    def get_aux_output(self):
        # 提供一个方法来获取辅助输出
        return self.contrast_logits

class CrossScaleAttention(nn.Module):
    def __init__(self, c1, n=1, c2=None):
        super().__init__()
        self.c1 = c1
        self.c2 = c2 or c1
        
        if self.c1 != self.c2:
            self.down_channel = nn.Conv2d(c1, self.c2, 1, bias=False)
        
        # 简化小目标增强分支
        self.small_defect_branch = nn.Sequential(
            nn.Conv2d(self.c2, self.c2 // 2, 1, bias=False),
            nn.BatchNorm2d(self.c2 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c2 // 2, self.c2 // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.c2 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c2 // 2, self.c2, 1, bias=False)
        )
        
        # 简化全局上下文分支
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c2, self.c2 // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c2 // 4, self.c2, 1, bias=False)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(self.c2 * 2, self.c2, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 可学习的残差权重
        self.alpha = nn.Parameter(torch.tensor(0.4))
    
    def forward(self, x):
        identity = x
        
        if hasattr(self, 'down_channel'):
            x = self.down_channel(x)
            identity = x
        
        # 小目标特征处理
        small_defect_feat = self.small_defect_branch(x)
        
        # 全局上下文处理
        global_feat = self.global_context(x)
        global_feat = F.interpolate(global_feat, 
                                  size=(small_defect_feat.size(2), small_defect_feat.size(3)), 
                                  mode='bilinear', 
                                  align_corners=False)
        
        # 特征融合
        fused = torch.cat([small_defect_feat, global_feat], dim=1)
        attention = self.fusion(fused)
        
        # 应用注意力
        enhanced_feat = x * attention
        
        # 使用可学习的残差连接
        return enhanced_feat * self.alpha + identity * (1 - self.alpha)
    
    def __deepcopy__(self, memo):
        # 创建一个新的模块实例
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # 复制所有状态
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        
        return result
        
    

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))  # 初始化为1而不是随机值可能更稳定
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta
    

class SRU(nn.Module):
    def __init__(self, 
                 oup_channels: int, 
                 group_num: int = 16, 
                 gate_treshold: float = 0.5, 
                 torch_gn: bool = False):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=group_num, num_channels=oup_channels) if torch_gn else GroupBatchnorm2d(c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        # 确保 gamma 是正数以避免除零错误
        w_gamma = torch.abs(self.gn.gamma) / (torch.sum(torch.abs(self.gn.gamma)) + 1e-10)
        reweights = self.sigmoid(gn_x * w_gamma)

        # 门控机制
        info_mask = reweights >= self.gate_treshold
        noninfo_mask = reweights < self.gate_treshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)
    

class CRU(nn.Module):
    def __init__(self, 
                 op_channel: int, 
                 alpha: float = 1/2, 
                 squeeze_radio: int = 2, 
                 group_size: int = 2, 
                 group_kernel_size: int = 3):
        super().__init__()

        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, max(1, up_channel // squeeze_radio), kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, max(1, low_channel // squeeze_radio), kernel_size=1, bias=False)

        # 上层特征转换
        self.GWC = nn.Conv2d(
            max(1, up_channel // squeeze_radio), 
            op_channel, 
            kernel_size=group_kernel_size, 
            stride=1, 
            padding=group_kernel_size // 2, 
            groups=min(group_size, max(1, up_channel // squeeze_radio))
        )
        self.PWC1 = nn.Conv2d(max(1, up_channel // squeeze_radio), op_channel, kernel_size=1, bias=False)

        # 下层特征转换
        self.PWC2 = nn.Conv2d(
            max(1, low_channel // squeeze_radio), 
            op_channel - max(1, low_channel // squeeze_radio), 
            kernel_size=1, 
            bias=False
        )
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 分割输入特征
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # 上层特征转换
        Y1 = self.GWC(up) + self.PWC1(up)

        # 下层特征转换
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        # 特征融合
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2
    

class SCConv(nn.Module):
    """
    SCConv模块 - 适配YOLO框架的实现
    
    参数:
        c1 (int): 输入通道数
        c2 (int): 输出通道数，默认等于c1
        group_num (int): SRU中的组数，默认为16
        gate_treshold (float): SRU中的门控阈值，默认为0.5
        alpha (float): CRU中的通道分割比例，默认为0.5
        squeeze_radio (int): CRU中的通道压缩比例，默认为2
        group_size (int): CRU中的组大小，默认为2
        group_kernel_size (int): CRU中的组卷积核大小，默认为3
    """
    def __init__(self, 
                 c1, 
                 c2=None,
                 group_num=16, 
                 gate_treshold=0.5, 
                 alpha=0.5, 
                 squeeze_radio=2, 
                 group_size=2, 
                 group_kernel_size=3):
        super().__init__()
        c2 = c2 if c2 is not None else c1
        
        # 如果输入通道数不等于输出通道数，添加1x1卷积进行调整
        self.conv = None
        if c1 != c2:
            self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.SRU = SRU(c2, group_num=group_num, gate_treshold=gate_treshold)
        self.CRU = CRU(c2, alpha=alpha, squeeze_radio=squeeze_radio, 
                      group_size=group_size, group_kernel_size=group_kernel_size)

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.SRU(x)
        x = self.CRU(x)
        return x


class DefectDetect:
    def __init__(self):
        pass