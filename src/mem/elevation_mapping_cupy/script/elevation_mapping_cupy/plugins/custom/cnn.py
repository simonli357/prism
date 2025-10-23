import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .color_mapping import SEM_CHANNELS
except ImportError:
    from color_mapping import SEM_CHANNELS

class ResidualBlock(nn.Module):
    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        p = dilation
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=p, dilation=dilation, bias=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=p, dilation=dilation, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)

class ASPPLite(nn.Module):
    def __init__(self, ch: int, out_ch: int, rates=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(nn.Conv2d(ch, out_ch, kernel_size=1, padding=0, bias=True))
            else:
                self.branches.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=True))
        self.proj = nn.Conv2d(out_ch * len(rates), out_ch, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        y = torch.cat(feats, dim=1)
        return self.act(self.proj(y))

class CNNCorrectionNetSingle(nn.Module):
    def __init__(
        self,
        in_ch_per_frame: int,       # C = SEM_CHANNELS + 1 (+1 if mask)
        base: int = 64,
        blocks_stage1: int = 4,
        blocks_stage2: int = 6,
        aspp_rates=(1, 2, 4),
        use_identity_correction: bool = True,
        return_edit: bool = False,  # optional: return edit prob for debugging/regularization
    ):
        super().__init__()
        self.use_identity = use_identity_correction
        self.return_edit = return_edit

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch_per_frame, base, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            ResidualBlock(base, dilation=1),
        )

        self.s1_in   = nn.Conv2d(base, base, kernel_size=3, padding=1)
        self.s1_body = nn.Sequential(*[ResidualBlock(base, d) for d in [1, 2, 1, 2][:blocks_stage1]])
        self.s1_aspp = ASPPLite(base, base, rates=aspp_rates)
        self.s1_head = nn.Conv2d(base, 1, kernel_size=1)   # edit logits
        nn.init.constant_(self.s1_head.bias, -2.0)         # start with few edits: sigmoid(-2)≈0.12

        self.s2_in   = nn.Conv2d(base + 1, base, kernel_size=3, padding=1)
        self.s2_aspp = ASPPLite(base, base, rates=aspp_rates)
        dilations    = [1, 1, 2, 2, 4, 4][:blocks_stage2]
        self.s2_body = nn.Sequential(*[ResidualBlock(base, d) for d in dilations])

        self.head_sem  = nn.Conv2d(base, SEM_CHANNELS, kernel_size=1)
        self.head_elev = nn.Conv2d(base, 1, kernel_size=1)
        nn.init.zeros_(self.head_sem.weight);  nn.init.zeros_(self.head_sem.bias)
        nn.init.zeros_(self.head_elev.weight); nn.init.zeros_(self.head_elev.bias)

    def forward(self, x):
        if x.dim() == 5:   # [B,T,C,H,W] -> use last frame
            x_last = x[:, -1]
        else:
            x_last = x

        sem_in   = x_last[:, :SEM_CHANNELS]                     # [B,14,H,W] (one-hot/prob)
        elev_in  = x_last[:, SEM_CHANNELS:SEM_CHANNELS+1]       # [B,1,H,W]

        f0 = self.stem(x_last)                                  # [B,base,H,W]

        f1 = F.relu(self.s1_in(f0), inplace=True)
        f1 = self.s1_body(f1)
        f1 = self.s1_aspp(f1)
        edit_logits = self.s1_head(f1)                          # [B,1,H,W]
        edit_prob   = torch.sigmoid(edit_logits)                # [0,1]

        f2_in = torch.cat([f0, edit_prob], dim=1)
        f2 = F.relu(self.s2_in(f2_in), inplace=True)
        f2 = self.s2_aspp(f2)
        f2 = self.s2_body(f2)

        d_sem  = self.head_sem(f2) * edit_prob                  # gate residuals
        d_elev = self.head_elev(f2) * edit_prob

        if self.use_identity:
            sem_logits_in  = torch.log(sem_in.clamp_min(1e-6))  # logit of one-hot/prob
            sem_logits_out = sem_logits_in + d_sem
            elev_out       = elev_in + d_elev
        else:
            sem_logits_out = d_sem
            elev_out       = d_elev

        out = torch.cat([sem_logits_out, elev_out], dim=1)
        if self.return_edit:
            return out, edit_prob
        return out
