import json
import os
import pathlib
import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.model_management as mm


# ── helpers ───────────────────────────────────────────────────────────────────

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """PIL Image → ComfyUI IMAGE tensor (1, H, W, C) float32 [0, 1]"""
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ── constants ─────────────────────────────────────────────────────────────────

QUALITY_SUFFIXES = {
    "none":           "",
    "photorealistic": ", RAW photo, photorealistic, high detail, 8k resolution, sharp focus, DSLR",
    "cinematic":      ", cinematic, dramatic lighting, film grain, anamorphic lens, movie still",
    "artistic":       ", artstation, concept art, intricate detail, vibrant colors, professional illustration",
    "minimalist":     ", clean composition, minimal, professional, high contrast",
}

NEGATIVE_PRESETS = {
    "standard": (
        "blurry, low quality, low resolution, jpeg artifacts, ugly, deformed, "
        "bad anatomy, watermark, text, logo, signature, extra limbs"
    ),
    "photo":    (
        "illustration, painting, drawing, art, cartoon, anime, cgi, render, "
        "blurry, low quality, watermark, text"
    ),
    "art":      "photo, realistic, 3d render, blurry, low quality, watermark, text",
    "none":     "",
}

# Cosmos3 supported resolutions (w, h) — 16:9, 1:1, 4:3, 3:4, 9:16 at 256/480/720p
RESOLUTIONS = {
    "1280x720 (16:9 HD)":      (1280, 720),
    "854x480  (16:9 480p)":    (854,  480),
    "456x256  (16:9 256p)":    (456,  256),
    "960x960  (1:1)":          (960,  960),
    "480x480  (1:1 small)":    (480,  480),
    "960x720  (4:3)":          (960,  720),
    "720x960  (3:4)":          (720,  960),
    "720x1280 (9:16 vertical)":(720, 1280),
    "custom":                  None,
}


# ── nodes ─────────────────────────────────────────────────────────────────────

class Cosmos3ModelLoader:
    """
    Downloads (first run) and loads Cosmos3-Nano as a diffusers DiffusionPipeline.

    Model is stored under ComfyUI's models/diffusion_models/Cosmos3-Nano/.
    Requires diffusers installed from git HEAD (see requirements.txt).
    Only bfloat16 is officially supported by NVIDIA.
    """

    MODEL_OPTIONS = ["nvidia/Cosmos3-Nano", "custom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODEL_OPTIONS, {"default": "nvidia/Cosmos3-Nano"}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "multiline": False,
                                           "tooltip": "Local directory or HF repo ID when model=custom"}),
            },
        }

    RETURN_TYPES = ("COSMOS3_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "Cosmos3"

    def load(self, model, custom_path=""):
        # ── import cosmos3 plugins ────────────────────────────────────────────
        try:
            import transformers_cosmos3  # noqa: F401 — requires transformers>=4.57
        except ImportError:
            raise ImportError(
                "[Cosmos3] transformers-cosmos3 not found. "
                "Install: pip install 'transformers-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/transformers-cosmos3'"
            )
        try:
            from diffusers_cosmos3 import Cosmos3OmniDiffusersPipeline
            import diffusers_cosmos3 as _dc3_pkg
        except ImportError:
            raise ImportError(
                "[Cosmos3] diffusers-cosmos3 not found. "
                "Install: pip install 'diffusers-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/diffusers-cosmos3'"
            )

        # ── sample_args patch ────────────────────────────────────────────────
        # The sample_args/*.json files that Cosmos3OmniDiffusersPipeline reads at
        # inference time may not be included in the pip install (packaging gap in
        # cosmos-framework as of June 2026).  Create them with correct defaults
        # from the README if they're missing.
        _sample_args_dir = pathlib.Path(_dc3_pkg.__file__).parent / "sample_args"
        _sample_args_dir.mkdir(exist_ok=True)
        # Default negative derived from NVIDIA's canonical negative_prompt.json:
        # covers the quality failure modes the model was trained to avoid.
        _COSMOS3_DEFAULT_NEG = (
            "blurry, low quality, jpeg artifacts, distorted features, unnatural proportions, "
            "floating subjects, broken geometry, visible compression artifacts, muddy textures, "
            "color bleeding, waxy skin, extra limbs, asymmetric face, teeth artifacts, "
            "flat lighting, no shadows, inconsistent light sources, flickering, temporal artifacts, "
            "shaky camera, rolling shutter, visible tiling, repeated textures, watermark, text, logo"
        )
        _mode_defaults = {
            "text2video":  {"guidance": 6.0, "num_steps": 10, "shift": 10.0,
                            "negative_prompt": _COSMOS3_DEFAULT_NEG,
                            "negative_prompt_keep_metadata": False},
            "image2video": {"guidance": 6.0, "num_steps": 10, "shift": 10.0,
                            "negative_prompt": _COSMOS3_DEFAULT_NEG,
                            "negative_prompt_keep_metadata": False},
        }
        for _mode, _defs in _mode_defaults.items():
            _p = _sample_args_dir / f"{_mode}.json"
            # Always overwrite so persistent containers pick up step-count changes
            _p.write_text(json.dumps(_defs, indent=2))
            print(f"[Cosmos3] Wrote sample_args/{_mode}.json (num_steps={_defs['num_steps']})")
        # ─────────────────────────────────────────────────────────────────────

        # ── RoPE 'default' patch ──────────────────────────────────────────────
        # transformers<4.57 may ship ROPE_INIT_FUNCTIONS without a 'default' key.
        # Patch the dict in-place before from_pretrained instantiates the transformer.
        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            if "default" not in ROPE_INIT_FUNCTIONS:
                try:
                    from transformers.modeling_rope_utils import _compute_default_rope_parameters
                    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
                    print("[Cosmos3] Patched ROPE_INIT_FUNCTIONS['default'] from transformers internals")
                except ImportError:
                    def _cosmos3_default_rope(config, device=None, seq_len=None, **kwargs):
                        base = getattr(config, "rope_theta", 10000.0)
                        head_dim = getattr(config, "head_dim",
                            getattr(config, "hidden_size", 4096) //
                            getattr(config, "num_attention_heads", 32))
                        dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
                        inv_freq = 1.0 / (base ** (
                            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
                        ))
                        return inv_freq, 1.0
                    ROPE_INIT_FUNCTIONS["default"] = _cosmos3_default_rope
                    print("[Cosmos3] Patched ROPE_INIT_FUNCTIONS['default'] with fallback implementation")
        except Exception as rope_err:
            print(f"[Cosmos3] Warning: could not patch ROPE_INIT_FUNCTIONS: {rope_err}")
        # ─────────────────────────────────────────────────────────────────────

        from huggingface_hub import snapshot_download

        if model == "custom":
            if not custom_path:
                raise ValueError("[Cosmos3] custom_path must be set when model='custom'")
            source = custom_path
            print(f"[Cosmos3] Loading from custom path: {source}")
        else:
            model_name = model.split("/")[-1]   # "Cosmos3-Nano"
            model_dir = os.path.join(folder_paths.models_dir, "diffusion_models", model_name)
            os.makedirs(model_dir, exist_ok=True)

            if not os.listdir(model_dir):
                print(f"[Cosmos3] First run — downloading {model} to {model_dir} (this will take a while)")
                snapshot_download(
                    repo_id=model,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                )
            else:
                print(f"[Cosmos3] Loading from cache: {model_dir}")

            source = model_dir

        # ── adaptive VRAM strategy ────────────────────────────────────────────
        # 16B model @ bfloat16 ≈ 32 GB weights.  Need headroom for activations.
        # Threshold: 40 GB — comfortably covers 6000 Ada (48 GB) but not 5090 (32 GB).
        _HIGH_VRAM_GB = 40.0
        _total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Cosmos3] GPU: {torch.cuda.get_device_name(0)} ({_total_vram_gb:.0f} GB)")

        if _total_vram_gb >= _HIGH_VRAM_GB:
            print("[Cosmos3] High-VRAM mode — device_map='balanced'")
            pipe = Cosmos3OmniDiffusersPipeline.from_pretrained(
                source,
                torch_dtype=torch.bfloat16,
                device_map="balanced",
            )
        else:
            print(f"[Cosmos3] Low-VRAM mode ({_total_vram_gb:.0f} GB) — "
                  "loading to CPU then enabling sequential offload (slower but fits)")
            pipe = Cosmos3OmniDiffusersPipeline.from_pretrained(
                source,
                torch_dtype=torch.bfloat16,
                # no device_map → loads to CPU
            )
            pipe.enable_sequential_cpu_offload()
        # ─────────────────────────────────────────────────────────────────────

        print("[Cosmos3] Pipeline loaded.")

        # ── tokenize_caption safety patch ─────────────────────────────────────
        # In some transformers/tokenizer configurations apply_chat_template(
        # tokenize=True) returns a formatted string instead of list[int].
        # The pipeline then iterates over that string char-by-char and passes
        # characters as token IDs → ValueError in torch.tensor().
        # Wrap the method to guarantee list[int] output.
        _orig_tc = pipe.tokenize_caption

        def _safe_tokenize_caption(caption, is_video=False, use_system_prompt=False):
            result = _orig_tc(caption, is_video=is_video, use_system_prompt=use_system_prompt)

            # Newer transformers returns BatchEncoding instead of list[int].
            # Iterating over BatchEncoding yields dict keys (strings), not token IDs.
            if hasattr(result, "input_ids"):
                ids = result.input_ids
                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                    ids = ids[0]
                elif hasattr(ids, "tolist"):
                    ids = ids.squeeze().tolist()
                result = ids

            if isinstance(result, str):
                result = pipe.text_tokenizer.encode(result, add_special_tokens=False)
            elif isinstance(result, list) and result and not isinstance(result[0], int):
                flat = []
                for item in result:
                    (flat.extend(item) if isinstance(item, list) else flat.append(int(item)))
                result = flat

            return result

        pipe.tokenize_caption = _safe_tokenize_caption

        # ── pack_input_sequence intercept ──────────────────────────────────────
        # Second line of defence: inspect and fix input_text_indexes right before
        # pack_input_sequence uses them, and log what we actually see so we can
        # diagnose the root cause.
        try:
            # IMPORTANT: pipeline.py does `from .sequence_packing import pack_input_sequence`
            # creating a LOCAL binding at import time.  Patching sequence_packing module
            # attribute doesn't affect that local name.  Patch the PIPELINE module instead.
            import diffusers_cosmos3.pipeline as _dc3_pl

            _orig_pis = _dc3_pl.pack_input_sequence

            def _safe_pack_input_sequence(*args, **kwargs):
                text_idx = kwargs.get("input_text_indexes")
                if text_idx is None and len(args) > 1:
                    text_idx = args[1]

                if text_idx is not None:
                    fixed = []
                    changed = False
                    for tokens in text_idx:
                        if hasattr(tokens, "input_ids"):
                            # BatchEncoding (newer transformers) — extract flat list[int]
                            ids = tokens.input_ids
                            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                                tokens = ids[0]
                            elif hasattr(ids, "tolist"):
                                tokens = ids.squeeze().tolist()
                            else:
                                tokens = ids
                            changed = True
                        elif isinstance(tokens, str):
                            tokens = pipe.text_tokenizer.encode(tokens)
                            changed = True
                        elif (isinstance(tokens, list) and tokens
                              and not isinstance(tokens[0], int)):
                            tokens = [int(t) for t in tokens]
                            changed = True
                        fixed.append(tokens)
                    if changed:
                        kwargs["input_text_indexes"] = fixed

                return _orig_pis(*args, **kwargs)

            _dc3_pl.pack_input_sequence = _safe_pack_input_sequence
        except Exception as _pe:
            print(f"[Cosmos3] Could not patch pack_input_sequence: {_pe}")
        # ─────────────────────────────────────────────────────────────────────

        return (pipe,)


# ──────────────────────────────────────────────────────────────────────────────

class Cosmos3PromptEnricher:
    """
    Locally enriches a plain-text prompt for Cosmos3 generation.
    No external API calls — quality boosting is done via curated suffix presets.

    Outputs a (prompt, negative_prompt) pair wired directly into the sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt":           ("STRING", {"multiline": True,  "default": ""}),
                "quality":          (list(QUALITY_SUFFIXES.keys()), {"default": "photorealistic"}),
                "negative_preset":  (list(NEGATIVE_PRESETS.keys()), {"default": "standard"}),
            },
            "optional": {
                "extra_positive":   ("STRING", {"multiline": False, "default": "",
                                                "tooltip": "Additional positive terms appended after quality suffix"}),
                "extra_negative":   ("STRING", {"multiline": False, "default": "",
                                                "tooltip": "Additional negative terms appended to the preset"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt")
    FUNCTION = "enrich"
    CATEGORY = "Cosmos3"

    def enrich(self, prompt, quality, negative_preset, extra_positive="", extra_negative=""):
        pos = prompt.strip() + QUALITY_SUFFIXES[quality]
        if extra_positive:
            pos += f", {extra_positive.strip()}"

        neg = NEGATIVE_PRESETS[negative_preset]
        if extra_negative:
            neg = f"{neg}, {extra_negative.strip()}" if neg else extra_negative.strip()

        return (pos, neg)


# ──────────────────────────────────────────────────────────────────────────────

class Cosmos3T2ISampler:
    """
    Cosmos3-Nano Text-to-Image (and Image-to-Image) sampler.

    Leave the 'image' socket unconnected for pure text-to-image.
    Connect an IMAGE to enable image-conditioned generation — the model
    uses its world-understanding to edit/reinterpret the scene based on
    the prompt while preserving spatial structure.

    Output: standard ComfyUI IMAGE tensor (1, H, W, C) float32.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":             ("COSMOS3_PIPELINE",),
                "prompt":               ("STRING", {"multiline": True, "forceInput": True}),
                "resolution":           (list(RESOLUTIONS.keys()), {"default": "1280x720 (16:9 HD)"}),
                "num_inference_steps":  ("INT",   {"default": 10,  "min": 1,   "max": 100, "step": 1}),
                "guidance_scale":       ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "seed":                 ("INT",   {"default": 0,   "min": 0,   "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "init_image":       ("IMAGE",  {"tooltip": "Connect for image-to-image mode."}),
                "strength":         ("FLOAT",  {"default": 0.75, "min": 0.05, "max": 1.0, "step": 0.05,
                                                "tooltip": "0 = keep input unchanged, 1 = full regeneration. "
                                                           "Ignored when init_image is not connected."}),
                "negative_prompt":  ("STRING", {"multiline": True, "forceInput": True}),
                "custom_width":     ("INT",    {"default": 1280, "min": 256, "max": 2048, "step": 16}),
                "custom_height":    ("INT",    {"default": 720,  "min": 256, "max": 2048, "step": 16}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Cosmos3"

    def generate(
        self,
        pipeline,
        prompt,
        resolution,
        num_inference_steps,
        guidance_scale,
        seed,
        init_image=None,
        strength=0.75,
        negative_prompt=None,
        custom_width=1280,
        custom_height=720,
    ):
        import inspect

        if resolution == "custom":
            w, h = custom_width, custom_height
        else:
            w, h = RESOLUTIONS[resolution]

        # ── auto-match resolution to input image aspect ratio (I2I) ──────────
        # When an init_image is provided and the user hasn't chosen "custom",
        # snap to the closest supported resolution so the input doesn't get
        # distorted by an incompatible aspect ratio crop.
        if init_image is not None and resolution != "custom":
            _in_w = init_image.shape[2]   # ComfyUI tensor: (B, H, W, C)
            _in_h = init_image.shape[1]
            _in_ar = _in_w / _in_h
            _best_res, _best_diff = (w, h), float("inf")
            for _rk, _rd in RESOLUTIONS.items():
                if _rd is None:
                    continue
                _res_ar = _rd[0] / _rd[1]
                _diff = abs(_res_ar - _in_ar)
                if _diff < _best_diff:
                    _best_diff, _best_res = _diff, _rd
            w, h = _best_res
            print(f"[Cosmos3 I2I] Input AR {_in_ar:.3f} → auto-matched resolution {w}×{h}")
        # ─────────────────────────────────────────────────────────────────────

        # Cosmos3 requires dimensions divisible by 16
        w = (w // 16) * 16
        h = (h // 16) * 16

        generator = torch.Generator(device=mm.get_torch_device()).manual_seed(seed)

        # ── update sample_args so the pipeline uses our step/guidance values ──
        # Cosmos3OmniDiffusersPipeline reads num_steps and guidance from
        # sample_args/text2video.json at call time, not from __call__ kwargs.
        try:
            import diffusers_cosmos3 as _dc3_sa
            _sa_dir = pathlib.Path(_dc3_sa.__file__).parent / "sample_args"
            for _mode in ("text2video", "image2video"):
                _p = _sa_dir / f"{_mode}.json"
                if _p.exists():
                    _d = json.loads(_p.read_text())
                    _d["num_steps"] = effective_steps
                    _d["guidance"] = guidance_scale
                    _p.write_text(json.dumps(_d, indent=2))
        except Exception as _e:
            print(f"[Cosmos3] Warning: could not update sample_args: {_e}")
        # ─────────────────────────────────────────────────────────────────────

        # Convert init_image ComfyUI tensor → PIL if provided (I2I mode)
        pil_init = None
        if init_image is not None:
            import numpy as np
            img_np = (init_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_init = Image.fromarray(img_np, mode="RGB")

        mode = "I2I" if pil_init is not None else "T2I"
        strength = float(strength) if pil_init is not None else 1.0

        # ── strength-based noise injection (I2I only) ─────────────────────────
        # Encode the input image to the pipeline's latent space, mix with random
        # noise at the given strength, and pass as the denoising starting point.
        # effective_steps = round(strength * steps) ensures we only run the portion
        # of the diffusion trajectory corresponding to our noise level.
        effective_steps = num_inference_steps
        if pil_init is not None and strength < 1.0:
            try:
                _dev = mm.get_torch_device()
                # 1. Load + preprocess to [3, 1, H, W] in [-1, 1] (matches pipeline internals)
                _img_tensor = pipeline._load_image_as_tensor(pil_init, h, w)  # [3, 1, H, W]
                _img_input  = _img_tensor.unsqueeze(0).to(_dev, torch.bfloat16)  # [1, 3, 1, H, W]
                # 2. Encode to latent space
                with torch.no_grad():
                    _x0 = pipeline.vision_tokenizer.encode(_img_input).contiguous().float().cpu()
                # 3. Mix: (1 - strength) * clean + strength * noise
                _noise      = torch.randn_like(_x0)
                _mixed      = (1.0 - strength) * _x0 + strength * _noise
                effective_steps = max(1, round(num_inference_steps * strength))
                print(f"[Cosmos3 I2I] strength={strength:.2f} → {effective_steps} effective steps, "
                      f"latent shape={list(_x0.shape)}")
            except Exception as _e:
                print(f"[Cosmos3 I2I] Warning: strength encoding failed ({_e}). "
                      f"Falling back to concept-level conditioning.")
                _mixed = None
        else:
            _mixed = None
        # ─────────────────────────────────────────────────────────────────────

        # Base kwargs — always supported by Cosmos3OmniDiffusersPipeline
        kwargs = dict(
            prompt=prompt,
            width=w,
            height=h,
            num_frames=1,               # single frame for both T2I and I2I
            generator=generator,
        )
        if pil_init is not None:
            kwargs["image"] = pil_init
        if _mixed is not None:
            kwargs["noises"] = [_mixed]  # list[Tensor] matching x0_tokens_vision shape
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        # Conditionally pass params that may not exist in all pipeline versions
        sig = inspect.signature(pipeline.__call__)
        if "num_inference_steps" in sig.parameters:
            kwargs["num_inference_steps"] = num_inference_steps
        if "guidance_scale" in sig.parameters:
            kwargs["guidance_scale"] = guidance_scale

        # Warn if sequential offload + high step count will likely exceed Graydient timeout.
        if not hasattr(pipeline, "_device_map_set") and effective_steps > 15:
            print(
                f"[Cosmos3] WARNING: {effective_steps} steps on sequential CPU offload "
                f"may exceed Graydient timeout. Recommend ≤ 15 steps total."
            )

        print(f"[Cosmos3 {mode}] {w}x{h} | seed={seed} | steps={effective_steps} | strength={strength:.2f}")
        result = pipeline(**kwargs)

        # Cosmos3OmniDiffusersPipeline returns list[Tensor[C, T, H, W]] in [0, 1]
        raw = result[0]
        if raw.ndim == 4:
            raw = raw[:, 0, :, :]   # [C, 1, H, W] → [C, H, W]
        # raw is [C, H, W], convert to ComfyUI [1, H, W, C] float32
        image_tensor = raw.float().clamp(0.0, 1.0).permute(1, 2, 0).unsqueeze(0)
        return (image_tensor,)


# ──────────────────────────────────────────────────────────────────────────────

class Cosmos3LoadImageFromURL:
    """
    Downloads an image from a URL and returns a ComfyUI IMAGE tensor.
    Use with Graydient's init_image_url field to feed reference images
    into the Cosmos3 T2I sampler for image-conditioned generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"
    CATEGORY = "Cosmos3"

    def load(self, url):
        import requests
        from io import BytesIO

        if not url or not url.strip():
            raise ValueError("[Cosmos3] LoadImageFromURL: url is empty")

        print(f"[Cosmos3] Downloading image from URL...")
        response = requests.get(url.strip(), timeout=60)
        response.raise_for_status()
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"[Cosmos3] Image loaded: {pil_image.size[0]}×{pil_image.size[1]}")
        return (pil2tensor(pil_image),)


# ── registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "Cosmos3ModelLoader":       Cosmos3ModelLoader,
    "Cosmos3PromptEnricher":    Cosmos3PromptEnricher,
    "Cosmos3T2ISampler":        Cosmos3T2ISampler,
    "Cosmos3LoadImageFromURL":  Cosmos3LoadImageFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Cosmos3ModelLoader":       "Cosmos3 Model Loader",
    "Cosmos3PromptEnricher":    "Cosmos3 Prompt Enricher",
    "Cosmos3T2ISampler":        "Cosmos3 T2I / I2I Sampler",
    "Cosmos3LoadImageFromURL":  "Cosmos3 Load Image From URL",
}
