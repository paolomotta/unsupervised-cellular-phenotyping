from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from hibou.hibou import CellViTHibou


@dataclass
class TokenOutput:
    tokens: torch.Tensor   # (1, T, D) on CPU, float32
    grid_hw: tuple         # (Gh, Gw)
    patch_size: int        # e.g., 16



class CellViTHibouWrapper:
    """
    Parameters
    ----------
    ckpt_path : str | Path
        Path to the CellViTHibou checkpoint (.pt / .pth).
    device : str
        'cuda' or 'cpu'.
    use_autocast : bool
        Use torch.autocast during forward for speed on GPU.
    input_size : int
        Resize input tiles to this size before the model (e.g., 256).
    patch_size : int
        ViT patch size (e.g., 16).
    magnification : int | float
        Passed to calculate_instance_map().
    """

    def __init__(self, ckpt_path, 
                 device="cuda", 
                 input_size=256, 
                 patch_size=16,
                 use_autocast=True, 
                 magnification=20, 
                 filter_bg=False):
        
        self.device = device
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.use_autocast = bool(use_autocast)
        self.magnification = magnification
        self.filter_bg = bool(filter_bg) # Whether to filter out background instances (class=0)

        # TODO: The arguments of the model can also be passed as arguments to initializer. Skipped for simplicity and because only one model available currently.
        self.model = CellViTHibou(
            hibou_path=None,
            num_nuclei_classes=6,
            num_tissue_classes=19,
            embed_dim=1024,
            input_channels=3,
            depth=24,
            num_heads=16,
            extract_layers=[6, 12, 18, 24],
        )
        # 2. Load state dict on CPU
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state)

        # 3. Cast to lower precision on CPU
        try:
            self.model = self.model.to(dtype=torch.bfloat16)
            self.model_dtype = torch.bfloat16
        except Exception:
            self.model = self.model.half()
            self.model_dtype = torch.float16

        self.model.eval()

        # 4. Move to device
        self.model = self.model.to(self.device, non_blocking=True)

        self.preproc = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.7068, 0.5755, 0.7220],
                        std=[0.1950, 0.2316, 0.1816]),
            T.Resize((self.input_size, self.input_size), antialias=True),
        ])
        self.autocast_dtype = self.model_dtype if (torch.cuda.is_available() and device.startswith("cuda")) else torch.float32


    @torch.inference_mode()
    def forward_tile(self, rgb_np):
        """
        Single-pass inference:
          - returns token embeddings and instance map from one forward call.
        """
        # H0, W0 = int(rgb_np.shape[0]), int(rgb_np.shape[1])
        if rgb_np.dtype != np.uint8:
            rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)

        x = self.preproc(rgb_np).unsqueeze(0).to(self.device)
        # Ensure input tensor matches model dtype
        x = x.to(dtype=self.model_dtype)

        class _NoCtx:
            def __enter__(self): return None
            def __exit__(self, *a): return False

        cm = torch.autocast("cuda", dtype=self.autocast_dtype) if self.use_autocast and self.device.startswith("cuda") else _NoCtx()

        with cm:
            out = self.model(x, retrieve_tokens=True)

        # ---- Tokens (B, D, Gh, Gw) -> (B, T, D) on CPU float32
        tok_4d = out["tokens"]                  # (1, D, Gh, Gw)
        B, D, Gh, Gw = tok_4d.shape
        tokens = tok_4d.permute(0, 2, 3, 1).reshape(B, Gh * Gw, D).float().cpu()
        tok = TokenOutput(tokens=tokens, grid_hw=(Gh, Gw), patch_size=self.patch_size)

        # ---- Nuclei maps → softmax (on CPU for safety/consistency) - CellViTHibou is returning logits
        for k in ("nuclei_binary_map", "nuclei_type_map"):
            if k in out and torch.is_tensor(out[k]):
                out[k] = F.softmax(out[k], dim=1).detach().float().cpu()

        # Move to CPU because to safely do np.concatenate in calculate_instance_map
        for k, v in list(out.items()):
            if torch.is_tensor(v):
                out[k] = v.detach().float().cpu()

        # ---- Instance map via Hibou API
        _, instance_types = self.model.calculate_instance_map(out, magnification=self.magnification)
        cells = instance_types[0] if isinstance(instance_types, (list, tuple)) else instance_types

        # Background classes are not useful for phenotyping, so we can filter them out here
        if self.filter_bg:
            cells = self.filter_background_class(cells, bg_class=0)

        return tok, cells



    def filter_background_class(self, cells, bg_class=0):
        """
        Remove instances predicted as background (class=bg_class) from cells dictionary.

        Parameters
        ----------
        cells : dict of dict
            As returned by forward_tile().
        bg_class : int
            Class index corresponding to background.

        Returns
        -------
        filtered_map : dict of dict
            Instance map with background instances removed.
        """
        filtered_map = {k: v for k, v in cells.items() if v.get("type", -1) != bg_class}
        return filtered_map