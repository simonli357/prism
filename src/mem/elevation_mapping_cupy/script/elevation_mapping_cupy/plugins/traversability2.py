import cupy as cp
import numpy as np
from typing import List, Optional, Tuple
from .plugin_manager import PluginBase


def _default_weights() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sensible default weights that match the expected shapes:
      conv1: (4, 1, 3, 3)  dilation=1
      conv2: (4, 1, 3, 3)  dilation=2 (same kernel size/shape)
      conv3: (4, 1, 3, 3)  dilation=3 (same kernel size/shape)
      conv_out: (1, 12, 1, 1)
    """
    # Basic 3x3 kernels: blur + Sobel-like edges + Laplacian-ish
    blur = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]], dtype=np.float32)
    blur = blur / blur.sum()

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = sobel_x.T

    lap = np.array([[0,  1, 0],
                    [1, -4, 1],
                    [0,  1, 0]], dtype=np.float32)

    # Stack four 3x3 filters -> (4, 1, 3, 3)
    kset = np.stack([blur, sobel_x, sobel_y, lap], axis=0)[:, None, :, :]  # (4,1,3,3)

    w1 = kset.copy().astype(np.float32)          # (4,1,3,3)
    w2 = kset.copy().astype(np.float32)          # (4,1,3,3)
    w3 = kset.copy().astype(np.float32)          # (4,1,3,3)

    # 12 input channels -> 1 output via 1x1 conv; start as equal-weight average
    w_out = np.ones((1, 12, 1, 1), dtype=np.float32) / 12.0

    return w1, w2, w3, w_out


def get_filter_torch(
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    w_out: np.ndarray,
    device: str = "cuda",
    use_bias: bool = False,
):
    import torch
    import torch.nn as nn

    class TraversabilityFilter(nn.Module):
        def __init__(self, w1, w2, w3, w_out, device="cuda", use_bias=False):
            super().__init__()
            print("TraversabilityFilter device", device)
            self.conv1 = nn.Conv2d(1, 4, 3, dilation=1, padding=0, bias=use_bias)
            self.conv2 = nn.Conv2d(1, 4, 3, dilation=2, padding=0, bias=use_bias)
            self.conv3 = nn.Conv2d(1, 4, 3, dilation=3, padding=0, bias=use_bias)
            self.conv_out = nn.Conv2d(12, 1, 1, bias=use_bias)

            self.conv1.weight = nn.Parameter(torch.from_numpy(w1).float())
            self.conv2.weight = nn.Parameter(torch.from_numpy(w2).float())
            self.conv3.weight = nn.Parameter(torch.from_numpy(w3).float())
            self.conv_out.weight = nn.Parameter(torch.from_numpy(w_out).float())

            if use_bias:
                nn.init.zeros_(self.conv1.bias)
                nn.init.zeros_(self.conv2.bias)
                nn.init.zeros_(self.conv3.bias)
                nn.init.zeros_(self.conv_out.bias)

        def forward(self, elevation_cupy: cp.ndarray) -> cp.ndarray:
            elevation_cupy = elevation_cupy.astype(cp.float32)

            elevation = torch.as_tensor(elevation_cupy, device=self.conv1.weight.device)

            with torch.no_grad():
                x = elevation.view(1, 1, elevation.shape[0], elevation.shape[1])
                out1 = self.conv1(x)                       # (1,4,H-2,W-2)
                out2 = self.conv2(x)                       # (1,4,H-4,W-4)
                out3 = self.conv3(x)                       # (1,4,H-6,W-6)

                out1 = out1[:, :, 2:-2, 2:-2]              # -> (1,4,H-6,W-6)
                out2 = out2[:, :, 1:-1, 1:-1]              # -> (1,4,H-6,W-6)

                feats = torch.cat((out1, out2, out3), dim=1)  # (1,12,H-6,W-6)

                y = self.conv_out(feats.abs())             # (1,1,H-6,W-6)
                y = torch.exp(-y)                          # higher = more traversable

                out_cupy = cp.asarray(y)

            return out_cupy  # (1,1,H-6,W-6)

    filt = TraversabilityFilter(w1, w2, w3, w_out, device=device, use_bias=use_bias)
    if device.startswith("cuda"):
        filt = filt.cuda().eval()
    else:
        filt = filt.cpu().eval()
    return filt


class Traversability2(PluginBase):
    def __init__(
        self,
        device: str = "cuda",
        use_bias: bool = False,
        w1: Optional[np.ndarray] = None,
        w2: Optional[np.ndarray] = None,
        w3: Optional[np.ndarray] = None,
        w_out: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__()
        if any(w is None for w in (w1, w2, w3, w_out)):
            w1d, w2d, w3d, w_outd = _default_weights()
            w1 = w1 if w1 is not None else w1d
            w2 = w2 if w2 is not None else w2d
            w3 = w3 if w3 is not None else w3d
            w_out = w_out if w_out is not None else w_outd

        self.device = device
        self.use_bias = use_bias
        self.filter = get_filter_torch(
            w1=w1, w2=w2, w3=w3, w_out=w_out, device=device, use_bias=use_bias
        )

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_layers: cp.ndarray,
        semantic_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        assert elevation_map.ndim == 3, f"elevation_map must be (N,H,W), got {elevation_map.shape}"
        elev = elevation_map[0]  # (H,W)

        out_cu = self.filter.forward(elev)

        inner = out_cu.reshape(out_cu.shape[-2], out_cu.shape[-1])

        H, W = elev.shape
        Hs, Ws = inner.shape
        assert Hs == H - 6 and Ws == W - 6, "Unexpected inner size from filter."

        traversability = cp.ones_like(elev, dtype=cp.float32)
        traversability[3:-3, 3:-3] = inner

        # min_val = cp.min(traversability)
        # max_val = cp.max(traversability)
        # avg_val = cp.mean(traversability)
        # median_val = cp.median(traversability)
        # print(f"Traversability2 Plugin: min={min_val:.4f}, max={max_val:.4f}, avg={avg_val:.4f}, median={median_val:.4f}")
        return traversability
