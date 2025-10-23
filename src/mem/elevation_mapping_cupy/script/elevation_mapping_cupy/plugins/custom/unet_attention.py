# attention_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .color_mapping import SEM_CHANNELS
except ImportError:
    from color_mapping import SEM_CHANNELS

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)

        g_conv_upsampled = F.interpolate(g_conv, size=x_conv.shape[2:], mode='bilinear', align_corners=False)

        gx_add = self.relu(g_conv_upsampled + x_conv)

        psi_out = self.psi(gx_add)
        return x * psi_out

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.block(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.block = ConvBlock(in_ch=out_ch * 2, out_ch=out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.reduce(x)
        
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, k=3):
        super().__init__()
        p = k // 2; self.in_ch = in_ch; self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, k, padding=p)
    def forward(self, x, h, c):
        if h is None:
            B, _, H, W = x.shape; device = x.device
            h = torch.zeros(B, self.hidden_ch, H, W, device=device)
            c = torch.zeros(B, self.hidden_ch, H, W, device=device)
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(z, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f * c + i * g; h = o * torch.tanh(c)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hidden_ch)
    def forward(self, x_seq):
        h = c = None
        for t in range(x_seq.shape[1]):
            h, c = self.cell(x_seq[:, t], h, c)
        return h

class AttentionUNetConvLSTM(nn.Module):
    def __init__(self, in_ch, base=32, out_ch=SEM_CHANNELS+1):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = Down(base, base * 2)
        self.enc3 = Down(base * 2, base * 4)
        self.enc4 = Down(base * 4, base * 8)
        self.lstm = ConvLSTM(in_ch=base * 8, hidden_ch=base * 8)
        self.up3 = Up(base * 8, base * 4)
        self.up2 = Up(base * 4, base * 2)
        self.up1 = Up(base * 2, base)
        self.att3 = AttentionGate(F_g=base * 8, F_l=base * 4, F_int=base * 4)
        self.att2 = AttentionGate(F_g=base * 4, F_l=base * 2, F_int=base * 2)
        self.att1 = AttentionGate(F_g=base * 2, F_l=base, F_int=base)
        self.out = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
      B, T, C, H, W = x.shape
      feats_for_lstm = []
      skip1_sum = skip2_sum = skip3_sum = None

      for t in range(T):
          xt = x[:, t]
          f1 = self.enc1(xt)   # H
          f2 = self.enc2(f1)   # ~H/2
          f3 = self.enc3(f2)   # ~H/4
          f4 = self.enc4(f3)   # ~H/8

          feats_for_lstm.append(f4)

          if skip1_sum is None:
              skip1_sum, skip2_sum, skip3_sum = f1, f2, f3
          else:
              skip1_sum = skip1_sum + f1
              skip2_sum = skip2_sum + f2
              skip3_sum = skip3_sum + f3

      skip1 = skip1_sum / T
      skip2 = skip2_sum / T
      skip3 = skip3_sum / T

      feats_for_lstm = torch.stack(feats_for_lstm, dim=1)
      bottleneck = self.lstm(feats_for_lstm)

      att_skip3 = self.att3(g=bottleneck, x=skip3)
      d3 = self.up3(bottleneck, att_skip3)

      att_skip2 = self.att2(g=d3, x=skip2)
      d2 = self.up2(d3, att_skip2)

      att_skip1 = self.att1(g=d2, x=skip1)
      d1 = self.up1(d2, att_skip1)

      return self.out(d1)
