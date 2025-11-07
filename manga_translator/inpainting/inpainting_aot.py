from typing import List, Optional
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from .inpainting_lama_mpe import LamaMPEInpainter

class AotInpainter(LamaMPEInpainter):
    _MODEL_MAPPING = {
        'model': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt',
            'hash': '878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f',
            'file': '.',
        },
        'onnx': {
            'url': 'https://github.com/frederik-uni/manga-image-translator-rust/releases/download/lama_aot/model.onnx',
            'hash': 'c5965aca4e5ffa8269051dca1fc30e379d2bded46e0a55366e299ade47086cfc',
            'file': 'lamaaotl.onnx',
        },
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists('inpainting.ckpt'):
            shutil.move('inpainting.ckpt', self._get_file_path('inpainting.ckpt'))
        super().__init__(*args, **kwargs)
    
    def _check_downloaded_map(self, map_key: str) -> bool:
        """如果ONNX模型存在，跳过PyTorch模型检查"""
        onnx_path = self._get_file_path('lamaaotl.onnx')
        if os.path.isfile(onnx_path):
            return True  # ONNX存在，不检查.ckpt
        return super()._check_downloaded_map(map_key)

    async def _load(self, device: str):
        self.device = device
        
        # ✅ CPU模式使用ONNX（解决虚拟内存泄漏，22MB小模型）
        if not device.startswith('cuda') and device != 'mps':
            try:
                import onnxruntime as ort
                onnx_path = self._get_file_path('lamaaotl.onnx')
                self.logger.info(f'使用ONNX模型（CPU优化，22MB小模型）: {onnx_path}')
                
                # 🔧 内存优化配置
                sess_options = ort.SessionOptions()
                sess_options.enable_mem_pattern = False  # 禁用内存模式优化可以减少内存占用
                sess_options.enable_cpu_mem_arena = False  # 禁用CPU内存池，按需分配
                
                self.session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']
                )
                self.backend = 'onnx'
                self.logger.info(f'ONNX Runtime版本: {ort.__version__}（内存优化模式）')
                return
            except Exception as e:
                self.logger.warning(f'ONNX加载失败，回退到PyTorch: {e}')
        
        # ✅ GPU模式或ONNX失败时使用PyTorch
        self.model = AOTGenerator()
        sd = torch.load(self._get_file_path('inpainting.ckpt'), map_location='cpu')
        self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        self.model.eval()
        self.backend = 'torch'
        if device.startswith('cuda') or device == 'mps':
            self.model.to(device)
    
    async def _unload(self):
        if hasattr(self, 'backend'):
            if self.backend == 'onnx':
                del self.session
            elif self.backend == 'torch':
                del self.model
        elif hasattr(self, 'model'):
            del self.model
    
    async def _infer(self, image: np.ndarray, mask: np.ndarray, config, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        # ✅ ONNX推理（AOT模型，2个输入，不含MPE）
        if hasattr(self, 'backend') and self.backend == 'onnx':
            return await self._infer_onnx_aot(image, mask, inpainting_size, verbose)
        
        # ✅ PyTorch推理（调用父类）
        return await super()._infer(image, mask, config, inpainting_size, verbose)
    
    async def _infer_onnx_aot(self, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        """ONNX推理方法（AOT模型，只需image和mask）"""
        import cv2
        from ..utils import resize_keep_aspect
        
        img_original = np.copy(image)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]
        
        height, width, c = image.shape
        if max(image.shape[0: 2]) > inpainting_size:
            image = resize_keep_aspect(image, inpainting_size)
            mask_resized = resize_keep_aspect(mask, inpainting_size)
            mask_original_resized = resize_keep_aspect(mask_original, inpainting_size)
        else:
            mask_resized = mask
            mask_original_resized = mask_original
        
        pad_size = 8
        h, w, c = image.shape
        new_h = h if h % pad_size == 0 else (pad_size - (h % pad_size)) + h
        new_w = w if w % pad_size == 0 else (pad_size - (w % pad_size)) + w
        
        # Padding
        img_pad = np.pad(image, ((0, new_h - h), (0, new_w - w), (0, 0)), mode='symmetric')
        # 根据 mask_original_resized 的维度决定 padding 参数
        if len(mask_original_resized.shape) == 3:
            mask_pad = np.pad(mask_original_resized, ((0, new_h - h), (0, new_w - w), (0, 0)), mode='symmetric')
        else:
            mask_pad = np.pad(mask_original_resized, ((0, new_h - h), (0, new_w - w)), mode='symmetric')
            mask_pad = mask_pad[:, :, None]  # 扩展为3维
        
        # 准备输入（0-1归一化）
        img = img_pad.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]  # [1, 3, H, W]
        
        mask_input = mask_pad.astype(np.float32)[:, :, 0:1]
        mask_input = np.transpose(mask_input, (2, 0, 1))[None, ...]  # [1, 1, H, W]
        
        # ONNX推理（只需2个输入：image和mask）
        ort_inputs = {
            'image': img.astype(np.float32),
            'mask': mask_input.astype(np.float32)
        }
        img_inpainted = self.session.run(None, ort_inputs)[0]
        
        # 后处理
        img_inpainted = np.transpose(img_inpainted[0], (1, 2, 0))  # [H, W, 3]
        img_inpainted = (img_inpainted * 255.).astype(np.uint8)
        
        # Remove padding
        img_inpainted = img_inpainted[:h, :w, :]
        
        # Resize back
        if max(height, width) > inpainting_size:
            img_inpainted = cv2.resize(img_inpainted, (width, height), interpolation=cv2.INTER_LINEAR)
            mask_original_resized = cv2.resize(mask_original_resized, (width, height), interpolation=cv2.INTER_LINEAR)
            if len(mask_original_resized.shape) == 2:
                mask_original_resized = mask_original_resized[:, :, None]
        
        ans = img_inpainted * mask_original_resized + img_original * (1 - mask_original_resized)
        
        # ✅ ONNX内存清理
        import gc
        del img, mask_input, ort_inputs, img_inpainted, img_original, mask_original, mask_original_resized
        gc.collect()
        
        return ans


def relu_nf(x):
    return F.relu(x) * 1.7139588594436646

def gelu_nf(x):
    return F.gelu(x) * 1.7015043497085571

def silu_nf(x):
    return F.silu(x) * 1.7881293296813965

class LambdaLayer(nn.Module):
    def __init__(self, f):
        super(LambdaLayer, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

class ScaledWSConv2d(nn.Conv2d):
    """2D Conv layer with Scaled Weight Standardization."""
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=0,
        dilation=1, groups=1, bias=True, gain=True,
        eps=1e-4):
        nn.Conv2d.__init__(self, in_channels, out_channels,
            kernel_size, stride,
            padding, dilation,
            groups, bias)
        #nn.init.kaiming_normal_(self.weight)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        # Epsilon, a small constant to avoid dividing by zero.
        self.eps = eps
    def get_weight(self):
        # Get Scaled WS weight OIHW;
        fan_in = np.prod(self.weight.shape[1:])
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
        scale = torch.rsqrt(torch.max(
            var * fan_in, torch.tensor(self.eps).to(var.device))) * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias,
            self.stride, self.padding,
            self.dilation, self.groups)

class ScaledWSTransposeConv2d(nn.ConvTranspose2d):
    """2D Transpose Conv layer with Scaled Weight Standardization."""
    def __init__(self, in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        gain=True,
        eps=1e-4):
        nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, 'zeros')
        #nn.init.kaiming_normal_(self.weight)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.in_channels, 1, 1, 1))
        else:
            self.gain = None
        # Epsilon, a small constant to avoid dividing by zero.
        self.eps = eps
    def get_weight(self):
        # Get Scaled WS weight OIHW;
        fan_in = np.prod(self.weight.shape[1:])
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
        scale = torch.rsqrt(torch.max(
            var * fan_in, torch.tensor(self.eps).to(var.device))) * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, x, output_size: Optional[List[int]] = None):
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose2d(x, self.get_weight(), self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

class GatedWSConvPadded(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride = 1, dilation = 1):
        super(GatedWSConvPadded, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.padding = nn.ReflectionPad2d(((ks - 1) * dilation) // 2)
        self.conv = ScaledWSConv2d(in_ch, out_ch, kernel_size = ks, stride = stride, dilation = dilation)
        self.conv_gate = ScaledWSConv2d(in_ch, out_ch, kernel_size = ks, stride = stride, dilation = dilation)

    def forward(self, x):
        x = self.padding(x)
        signal = self.conv(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return signal * gate * 1.8

class GatedWSTransposeConvPadded(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride = 1):
        super(GatedWSTransposeConvPadded, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = ScaledWSTransposeConv2d(in_ch, out_ch, kernel_size = ks, stride = stride, padding = (ks - 1) // 2)
        self.conv_gate = ScaledWSTransposeConv2d(in_ch, out_ch, kernel_size = ks, stride = stride, padding = (ks - 1) // 2)

    def forward(self, x):
        signal = self.conv(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return signal * gate * 1.8

class ResBlock(nn.Module):
    def __init__(self, ch, alpha = 0.2, beta = 1.0, dilation = 1):
        super(ResBlock, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.c1 = GatedWSConvPadded(ch, ch, 3, dilation = dilation)
        self.c2 = GatedWSConvPadded(ch, ch, 3, dilation = dilation)

    def forward(self, x):
        skip = x
        x = self.c1(relu_nf(x / self.beta))
        x = self.c2(relu_nf(x))
        x = x * self.alpha
        return x + skip

def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

class AOTBlock(nn.Module):
    def __init__(self, dim, rates = [2, 4, 8, 16]):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask

class ResBlockDis(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResBlockDis, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3 if stride == 1 else 4, stride=stride, padding=1)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.planes = planes
        self.in_planes = in_planes
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.Sequential(nn.AvgPool2d(2, 2), nn.Conv2d(in_planes, planes, kernel_size=1))
        elif in_planes != planes and stride == 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1))

    def forward(self, x):
        sc = self.shortcut(x)
        x = self.conv1(F.leaky_relu(self.bn1(x), 0.2))
        x = self.conv2(F.leaky_relu(self.bn2(x), 0.2))
        return sc + x
from torch.nn.utils import spectral_norm
class Discriminator(nn.Module):
    def __init__(self, in_ch = 3, in_planes = 64, blocks = [2, 2, 2], alpha = 0.2):
        super(Discriminator, self).__init__()
        self.in_planes = in_planes

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, in_planes, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_planes, in_planes*2, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_planes*2, in_planes*4, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_planes*4, in_planes*8, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class AOTGenerator(nn.Module):
    def __init__(self, in_ch = 4, out_ch = 3, ch = 32, alpha = 0.0):
        super(AOTGenerator, self).__init__()

        self.head = nn.Sequential(
            GatedWSConvPadded(in_ch, ch, 3, stride = 1),
            LambdaLayer(relu_nf),
            GatedWSConvPadded(ch, ch * 2, 4, stride = 2),
            LambdaLayer(relu_nf),
            GatedWSConvPadded(ch * 2, ch * 4, 4, stride = 2),
        )

        self.body_conv = nn.Sequential(*[AOTBlock(ch * 4) for _ in range(10)])

        self.tail = nn.Sequential(
            GatedWSConvPadded(ch * 4, ch * 4, 3, 1),
            LambdaLayer(relu_nf),
            GatedWSConvPadded(ch * 4, ch * 4, 3, 1),
            LambdaLayer(relu_nf),
            GatedWSTransposeConvPadded(ch * 4, ch * 2, 4, 2),
            LambdaLayer(relu_nf),
            GatedWSTransposeConvPadded(ch * 2, ch, 4, 2),
            LambdaLayer(relu_nf),
            GatedWSConvPadded(ch, out_ch, 3, stride = 1),
        )

    def forward(self, img, mask):
        x = torch.cat([mask, img], dim = 1)
        x = self.head(x)
        conv = self.body_conv(x)
        x = self.tail(conv)
        if self.training:
            return x
        else:
            return torch.clip(x, -1, 1)

def test():
    img = torch.randn(4, 3, 256, 256).cuda()
    mask = torch.randn(4, 1, 256, 256).cuda()
    net = AOTGenerator().cuda()
    y1 = net(img, mask)
    print(y1.shape)
