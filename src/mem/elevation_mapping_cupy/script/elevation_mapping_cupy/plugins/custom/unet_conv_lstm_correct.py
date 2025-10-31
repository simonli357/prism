import torch, torch.nn as nn, torch.nn.functional as F
try: 
    from .color_mapping import SEM_CHANNELS
except ImportError:
    from color_mapping import SEM_CHANNELS

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
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
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            diffY = skip.shape[-2] - x.shape[-2]
            diffX = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, k=3):
        super().__init__()
        p = k // 2
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, k, padding=p)
    def forward(self, x, h, c):
        if h is None:
            B, _, H, W = x.shape
            device = x.device
            h = torch.zeros(B, self.hidden_ch, H, W, device=device)
            c = torch.zeros(B, self.hidden_ch, H, W, device=device)
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(z, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
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

class UNetConvLSTMCorrection(nn.Module):
    def __init__(self, in_ch, base=64, out_ch=SEM_CHANNELS+1):
        super().__init__()
        assert out_ch == SEM_CHANNELS + 1, "This drop-in expects out_ch = SEM_CHANNELS + 1 (semantics + 1 elev)."

        self.enc1 = ConvBlock(in_ch, base)           # -> base, H,   W
        self.enc2 = Down(base, base * 2)             # -> 2b,  H/2, W/2
        self.enc3 = Down(base * 2, base * 4)         # -> 4b,  H/4, W/4

        self.lstm = ConvLSTM(in_ch=base * 4, hidden_ch=base * 4)

        self.up2 = Up(base * 4, base * 2)            # -> 2b,  H/2, W/2   (skip enc2)
        self.up1 = Up(base * 2, base)                # -> b,   H,   W    (skip enc1)

        self.sem_head = nn.Sequential(
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, SEM_CHANNELS, kernel_size=1)
        )

        elev_in_ch = base + (base * 2)  # d1 (b) + upsample(d2) (2b)
        self.elev_head = nn.Sequential(
            nn.Conv2d(elev_in_ch, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, kernel_size=1)
        )

        self.elev_dropout = nn.Dropout2d(p=0.1)

        last_sem_conv = self.sem_head[-1]
        if isinstance(last_sem_conv, nn.Conv2d):
            nn.init.zeros_(last_sem_conv.weight)
            nn.init.zeros_(last_sem_conv.bias)

        last_elev_conv = self.elev_head[-1]
        if isinstance(last_elev_conv, nn.Conv2d):
            nn.init.zeros_(last_elev_conv.weight)
            nn.init.zeros_(last_elev_conv.bias)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x_last = x[:, -1]
        sem_in = x_last[:, :SEM_CHANNELS]                 # [B, 14, H, W]
        elev_in = x_last[:, SEM_CHANNELS:SEM_CHANNELS+1]  # [B, 1, H, W]

        feats = []
        for t in range(T):
            xt = x[:, t]
            f1 = self.enc1(xt)         # H, W
            f2 = self.enc2(f1)         # H/2, W/2
            f3 = self.enc3(f2)         # H/4, W/4
            feats.append(f3)
            if t == T - 1:
                skip2, skip1 = f2, f1

        feats = torch.stack(feats, dim=1)            # [B, T, 4b, H/4, W/4]
        bottleneck = self.lstm(feats)                # [B, 4b, H/4, W/4]

        d2 = self.up2(bottleneck, skip2)             # [B, 2b, H/2, W/2]
        d1 = self.up1(d2,        skip1)              # [B, b,  H,   W]

        d_sem = self.sem_head(d1)                    # [B, K, H, W]
        # Convert input to logit space and apply correction
        sem_logits_in = torch.log(sem_in.clamp_min(1e-6))
        sem_logits_out = sem_logits_in + d_sem

        # --- Elevation Head (already corrective) ---
        d2_up = F.interpolate(d2, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        elev_in_features = torch.cat([d1, self.elev_dropout(d2_up)], dim=1)  # [B, b+2b, H, W]
        
        d_elev = self.elev_head(elev_in_features)    # [B, 1, H, W]
        elev_out = elev_in + d_elev

        out = torch.cat([sem_logits_out, elev_out], dim=1)
        return out