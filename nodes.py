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
    QUANT_OPTIONS = ["none", "int8", "int4"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODEL_OPTIONS, {"default": "nvidia/Cosmos3-Nano"}),
                "quantization": (cls.QUANT_OPTIONS, {"default": "int8",
                    "tooltip": (
                        "none  = BF16 (32 GB, needs 48 GB GPU or sequential offload)\n"
                        "int8  = INT8 weights (16 GB, fits 4090/5090 — RECOMMENDED)\n"
                        "int4  = INT4 weights ( 8 GB, fits 3090/smaller GPU)"
                    )}),
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

    def load(self, model, quantization="int8", custom_path=""):
        # ── import cosmos3 plugins ────────────────────────────────────────────
        # transformers_cosmos3 is no longer required now that Cosmos3OmniPipeline
        # comes from mainline diffusers (which uses plain transformers.AutoTokenizer,
        # not this plugin) — see the Cosmos3OmniDiffusersPipeline import note below.
        # Soft-import only: if a future dependency turns out to need it after all,
        # this degrades to a visible downstream error rather than a hard crash here.
        try:
            import transformers_cosmos3  # noqa: F401
            print("[Cosmos3] transformers_cosmos3 available (not required, but present)")
        except ImportError:
            pass
        try:
            from diffusers import Cosmos3OmniPipeline as Cosmos3OmniDiffusersPipeline
        except ImportError as _e:
            raise ImportError(
                "[Cosmos3] Cosmos3OmniPipeline not found in installed diffusers. "
                "Cosmos3 support was merged into mainline diffusers "
                "(src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py) — there is no "
                "separate 'diffusers-cosmos3' plugin package; that repo path never existed. "
                "Install diffusers from git HEAD: "
                "pip install git+https://github.com/huggingface/diffusers.git"
            ) from _e

        # ── sample_args patch: NO LONGER NEEDED ────────────────────────────────
        # This used to write sample_args/*.json files because the old standalone
        # diffusers-cosmos3 plugin read num_steps/guidance from disk instead of
        # __call__ kwargs. Mainline diffusers' Cosmos3OmniPipeline takes
        # num_inference_steps/guidance_scale directly as call arguments (see the
        # inspect.signature-gated kwargs building in Cosmos3T2VSampler/T2ISampler
        # below), so there's nothing to patch here anymore.
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
            # If custom_path looks like an HF repo id (not an existing local dir) and a
            # concept_mapping-staged copy already exists locally (same convention as the
            # non-custom branch below: models/diffusion_models/<repo_basename>/), prefer
            # that over re-resolving/downloading from the Hub. model_index.json presence
            # is the marker that a full staged copy is there, not a partial/failed one.
            if not os.path.isdir(custom_path):
                _staged_name = custom_path.split("/")[-1]
                _staged_dir = os.path.join(folder_paths.models_dir, "diffusion_models", _staged_name)
                if os.path.isfile(os.path.join(_staged_dir, "model_index.json")):
                    source = _staged_dir
                    print(f"[Cosmos3] Loading from concept_mapping-staged copy: {source}")
                else:
                    source = custom_path
                    print(f"[Cosmos3] Loading from custom path (no staged copy found, resolving via Hub): {source}")
            else:
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

        # ── quantization + adaptive VRAM strategy ────────────────────────────
        # Memory requirements:
        #   none → 32 GB BF16 → needs 40 GB+ for full GPU (device_map=balanced)
        #   int8 → 16 GB INT8 → fits 24 GB+ GPU with model_cpu_offload
        #   int4 →  8 GB INT4 → fits 12 GB+ GPU with model_cpu_offload
        # Compute stays BF16 throughout (weights-only quantization).
        _total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Cosmos3] GPU: {torch.cuda.get_device_name(0)} ({_total_vram_gb:.0f} GB) | quant={quantization}")

        # ── path 1: BF16 on a high-VRAM GPU (48 GB+) ─────────────────────────
        if quantization == "none" and _total_vram_gb >= 40.0:
            print("[Cosmos3] BF16 high-VRAM — device_map='balanced'")
            pipe = Cosmos3OmniDiffusersPipeline.from_pretrained(
                source, torch_dtype=torch.bfloat16, device_map="balanced"
            )

        # ── path 2: quantized (int8 / int4) ──────────────────────────────────
        elif quantization in ("int8", "int4"):
            pipe = Cosmos3OmniDiffusersPipeline.from_pretrained(
                source, torch_dtype=torch.bfloat16
            )
            _quant_applied = False
            _transformer_headroom = {"int8": 20.0, "int4": 12.0}
            _has_headroom = _total_vram_gb >= _transformer_headroom.get(quantization, 20.0)

            # torchao's quantized tensor subclass cannot survive ANY subsequent
            # .to(device) call in this environment — confirmed live 2026-09-30
            # (Cosmos3-Edge, RTX 5090): plain pipe.to("cuda"), accelerate's
            # enable_model_cpu_offload(), and enable_sequential_cpu_offload() all
            # crashed identically inside torchao/utils.py's storage-aliasing dispatch
            # ("Attempted to set the storage of a tensor on device X to a storage on
            # different device Y"). Root cause: torchao's compiled extensions
            # (_C_mxfp8/_C_cutlass) failed to load ("Unable to import torchao Tensor
            # objects"), leaving its device-move dispatch broken. Quantizing BEFORE
            # the device move (the original order) therefore always crashed on the
            # move that followed, regardless of which offload strategy was tried.
            #
            # Fix: when there's headroom, move to GPU FIRST as plain BF16 (a device
            # move that's proven to work), then quantize the already GPU-resident
            # transformer in place — no further device move is needed afterward.
            if _has_headroom:
                pipe.to("cuda")
                try:
                    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
                    _qfn = Int8WeightOnlyConfig() if quantization == "int8" else Int4WeightOnlyConfig()
                    quantize_(pipe.transformer, _qfn)
                    _quant_applied = True
                    print(f"[Cosmos3] {quantization.upper()} applied on GPU-resident transformer "
                          f"— ~{'16' if quantization == 'int8' else '8'} GB, no further device move needed")
                except Exception as _qe:
                    print(f"[Cosmos3] WARNING: {quantization.upper()} failed ({type(_qe).__name__}: {_qe!s:.120}) "
                          f"— continuing BF16 (already resident on GPU)")
                    quantization = "none"
            else:
                # Not enough VRAM for full BF16 residency even before quantizing —
                # quantize on CPU first to shrink the footprint, then offload. This
                # path can still hit the torchao device-move issue above; it's the
                # only option available when genuinely short on VRAM.
                try:
                    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
                    _qfn = Int8WeightOnlyConfig() if quantization == "int8" else Int4WeightOnlyConfig()
                    quantize_(pipe.transformer, _qfn)
                    _quant_applied = True
                    print(f"[Cosmos3] {quantization.upper()} applied — transformer ~"
                          f"{'16' if quantization == 'int8' else '8'} GB")
                except Exception as _qe:
                    print(f"[Cosmos3] WARNING: {quantization.upper()} failed ({type(_qe).__name__}: {_qe!s:.120}) "
                          f"— using BF16 + sequential offload")
                    quantization = "none"
                print("[Cosmos3] Sequential CPU offload")
                pipe.enable_sequential_cpu_offload()

        # ── path 3: BF16 on a low/mid VRAM GPU — sequential offload ──────────
        else:
            print(f"[Cosmos3] BF16 sequential CPU offload ({_total_vram_gb:.0f} GB GPU)")
            pipe = Cosmos3OmniDiffusersPipeline.from_pretrained(
                source, torch_dtype=torch.bfloat16
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
        # This patch targeted a bug in the old standalone diffusers-cosmos3 plugin;
        # mainline diffusers' Cosmos3OmniPipeline is a different implementation and
        # may not expose tokenize_caption at all, so this is best-effort.
        try:
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
        except AttributeError:
            print("[Cosmos3] pipeline has no tokenize_caption to patch — skipping (expected on mainline diffusers)")

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


# ──────────────────────────────────────────────────────────────────────────────

class Cosmos3T2VSampler:
    """
    Cosmos3-Nano Text-to-Video (and Image-to-Video) sampler.

    Leave init_image unconnected for T2V.
    Connect init_image for I2V — the model animates from the input image,
    with the first frame anchored to it and subsequent frames generated
    by the world model based on the prompt.

    Outputs the first frame as a ComfyUI IMAGE (for preview + Graydient capture)
    and saves the full video as MP4 to the output directory.

    num_frames tips:
      • Stick to 4k+1 values (5, 9, 17, 33, 65, 129, 189, 257...) for clean temporal
        compression.
      • Start with 33 frames (~1.4 s) to probe memory; scale up from there.
      • 189 was a leftover Nano-era OOM guess, not a real model/API ceiling — the
        pipeline uses RoPE (computed from sequence shape), not a fixed position
        table, so nothing in the code hard-stops at 189. NVIDIA's own docs list
        different validated ranges per model tier (Edge: 50-150, Super-4Step: up
        to 400) but those are "what we tested," not enforced maximums. Past that
        range you're extrapolating RoPE beyond its trained distribution — quality
        is unverified, not guaranteed to error.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":            ("COSMOS3_PIPELINE",),
                "prompt":              ("STRING", {"multiline": True, "forceInput": True}),
                "num_frames":          ("INT",   {"default": 33,  "min": 5,   "max": 4001, "step": 4,
                                                  "tooltip": "4k+1 values only (5/9/17/33/65/129/189/257/...). "
                                                             "No hard model ceiling — NVIDIA validated up to ~400 "
                                                             "for Super-4Step; past that is untested territory."}),
                "resolution":          (list(RESOLUTIONS.keys()), {"default": "1280x720 (16:9 HD)"}),
                "fps":                 ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "num_inference_steps": ("INT",   {"default": 10,  "min": 1,   "max": 100, "step": 1}),
                "guidance_scale":      ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "seed":                ("INT",   {"default": 0,   "min": 0,   "max": 0xFFFFFFFFFFFFFFFF}),
                "filename_prefix":     ("STRING", {"default": "cosmos3/t2v"}),
            },
            "optional": {
                "init_image":      ("IMAGE",  {"tooltip": "Connect for I2V mode — animates from this image"}),
                "negative_prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "custom_width":    ("INT",    {"default": 1280, "min": 256, "max": 2048, "step": 16}),
                "custom_height":   ("INT",    {"default": 720,  "min": 256, "max": 2048, "step": 16}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("first_frame", "video_path")
    FUNCTION = "generate"
    CATEGORY = "Cosmos3"

    def generate(
        self,
        pipeline,
        prompt,
        num_frames,
        resolution,
        fps,
        num_inference_steps,
        guidance_scale,
        seed,
        filename_prefix="cosmos3/t2v",
        init_image=None,
        negative_prompt=None,
        custom_width=1280,
        custom_height=720,
    ):
        import inspect

        if resolution == "custom":
            w, h = custom_width, custom_height
        else:
            w, h = RESOLUTIONS[resolution]
        w = (w // 16) * 16
        h = (h // 16) * 16

        generator = torch.Generator(device=mm.get_torch_device()).manual_seed(seed)

        # Update sample_args with step/guidance values
        try:
            import diffusers_cosmos3 as _dc3_v
            _sa_dir_v = pathlib.Path(_dc3_v.__file__).parent / "sample_args"
            for _m in ("text2video", "image2video"):
                _pv = _sa_dir_v / f"{_m}.json"
                if _pv.exists():
                    _dv = json.loads(_pv.read_text())
                    _dv["num_steps"] = num_inference_steps
                    _dv["guidance"] = guidance_scale
                    _pv.write_text(json.dumps(_dv, indent=2))
        except Exception as _e:
            print(f"[Cosmos3 T2V] Warning: could not update sample_args: {_e}")

        # Convert init_image ComfyUI tensor → PIL if provided (I2V mode)
        pil_init = None
        if init_image is not None:
            _img_np = (init_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_init = Image.fromarray(_img_np, mode="RGB")

        mode = "I2V" if pil_init is not None else "T2V"

        kwargs = dict(
            prompt=prompt,
            width=w,
            height=h,
            num_frames=num_frames,
            fps=float(fps),
            generator=generator,
        )
        if pil_init is not None:
            kwargs["image"] = pil_init
            # Cosmos3OmniPipeline anchors frame 0 to `image` automatically for I2V.
            # condition_frame_indexes_vision is a real kwarg but only documented for
            # the `video` (vid2vid) input path, which this node doesn't expose — no
            # equivalent is needed/accepted for plain image conditioning.
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        sig = inspect.signature(pipeline.__call__)
        if "num_inference_steps" in sig.parameters:
            kwargs["num_inference_steps"] = num_inference_steps
        if "guidance_scale" in sig.parameters:
            kwargs["guidance_scale"] = guidance_scale
        if "output_type" in sig.parameters:
            # Default output_type is "pil" (list[PIL.Image]) on Cosmos3OmniPipeline —
            # request a tensor directly so the .float()/.permute() below has something
            # to operate on.
            kwargs["output_type"] = "pt"

        print(f"[Cosmos3 {mode}] {w}×{h} | {num_frames} frames @ {fps:.0f}fps | "
              f"seed={seed} | steps={num_inference_steps}")
        result = pipeline(**kwargs)

        # Cosmos3OmniPipeline with output_type="pt" returns video as Tensor[T, C, H, W]
        # in [0, 1] (frames first — NOT [C, T, H, W]), per its own docstring.
        frames_tchw = result[0].float().clamp(0.0, 1.0)  # [T, C, H, W]
        frames_thwc = frames_tchw.permute(0, 2, 3, 1)     # [T, H, W, C]

        # ── save MP4 ──────────────────────────────────────────────────────────
        video_path = ""
        try:
            import imageio.v3 as iio
            frames_np = (frames_thwc.cpu().numpy() * 255).astype(np.uint8)

            _safe = filename_prefix.strip("/\\")
            _save_dir = os.path.join(
                folder_paths.get_output_directory(),
                os.path.dirname(_safe) or "cosmos3/t2v"
            )
            os.makedirs(_save_dir, exist_ok=True)
            _base = os.path.basename(_safe) or "t2v"
            _fname = f"{_base}_{seed}_{num_frames}f.mp4"
            video_path = os.path.join(_save_dir, _fname)

            iio.imwrite(video_path, frames_np, fps=fps, codec="h264", quality=8)
            print(f"[Cosmos3 T2V] Saved {num_frames} frames → {video_path}")
        except Exception as _ve:
            print(f"[Cosmos3 T2V] Warning: MP4 save failed ({_ve})")

        # Return first frame as IMAGE for preview / Graydient capture
        first_frame = frames_thwc[0:1]  # [1, H, W, C]
        return (first_frame, video_path)


# ──────────────────────────────────────────────────────────────────────────────

class Cosmos3T2VChunkedSampler:
    """
    Long-form Cosmos3 T2V/I2V by chunked video-to-video continuation, run inside a
    single node so it can loop until a wall-clock budget is hit (for one Graydient
    job's ~380-400s timeout) rather than needing external multi-node/multi-job
    chaining, which ComfyUI has no native loop primitive for anyway.

    How continuation works (from diffusers' Cosmos3OmniPipeline.prepare_latents):
    each chunk after the first passes the previous chunk's tail frames as `video`,
    with condition_video_keep="first" (the supplied frames are treated as the START
    of this chunk's window and kept clean) and condition_frame_indexes_vision=(0,1)
    (the pipeline's own default — keeps the first `max(indices)*temporal_compression+1`
    raw frames, i.e. 5 frames when the VAE's temporal_compression is 4). The model
    then generates everything after those anchor frames for the rest of this chunk's
    num_frames. The output always reproduces the anchor frames at the start, so they
    are trimmed off before appending to the accumulated result (they'd otherwise
    duplicate the tail already in the previous chunk's output).

    Untested end-to-end as of first write — this is the first live test of chunked
    continuation on Cosmos3. If chunk boundaries show visible discontinuity/drift,
    that's a real finding about the technique, not necessarily a bug in this node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":            ("COSMOS3_PIPELINE",),
                "prompt":              ("STRING", {"multiline": True, "forceInput": True}),
                "chunk_frames":        ("INT",   {"default": 33, "min": 5, "max": 189, "step": 4,
                                                  "tooltip": "Frames generated per chunk (4k+1 values). "
                                                             "Kept independent from the total-length ceiling — "
                                                             "this is per-chunk VRAM/time cost, not total video length."}),
                "max_total_frames":    ("INT",   {"default": 100000, "min": 9, "max": 1000000, "step": 1,
                                                  "tooltip": "Hard safety ceiling on total accumulated frames. "
                                                             "In practice max_seconds will almost always hit first."}),
                "max_seconds":         ("FLOAT", {"default": 270.0, "min": 5.0, "max": 3600.0, "step": 5.0,
                                                  "tooltip": "Wall-clock budget for the generation LOOP only — "
                                                             "does not include pip install/ComfyUI startup, which "
                                                             "eat into the same ~380-400s Graydient job timeout "
                                                             "before this node even starts. Default leaves ~110s "
                                                             "buffer for that platform overhead + MP4 encode/save."}),
                "resolution":          (list(RESOLUTIONS.keys()), {"default": "854x480  (16:9 480p)"}),
                "fps":                 ("FLOAT", {"default": 16.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "num_inference_steps": ("INT",   {"default": 10,  "min": 1,   "max": 100, "step": 1}),
                "guidance_scale":      ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "seed":                ("INT",   {"default": 0,   "min": 0,   "max": 0xFFFFFFFFFFFFFFFF}),
                "filename_prefix":     ("STRING", {"default": "cosmos3/t2v_long"}),
            },
            "optional": {
                "init_image":      ("IMAGE",  {"tooltip": "Optional — anchors chunk 0 as I2V. Omit for pure T2V."}),
                "negative_prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "custom_width":    ("INT",    {"default": 832, "min": 256, "max": 2048, "step": 16}),
                "custom_height":   ("INT",    {"default": 480, "min": 256, "max": 2048, "step": 16}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("first_frame", "video_path", "total_frames", "elapsed_seconds")
    FUNCTION = "generate"
    CATEGORY = "Cosmos3"

    def generate(
        self,
        pipeline,
        prompt,
        chunk_frames,
        max_total_frames,
        max_seconds,
        resolution,
        fps,
        num_inference_steps,
        guidance_scale,
        seed,
        filename_prefix="cosmos3/t2v_long",
        init_image=None,
        negative_prompt=None,
        custom_width=832,
        custom_height=480,
    ):
        import inspect
        import time

        if resolution == "custom":
            w, h = custom_width, custom_height
        else:
            w, h = RESOLUTIONS[resolution]
        w = (w // 16) * 16
        h = (h // 16) * 16

        # Latent-space conditioning window (pipeline's own default). Converted to a
        # raw-frame count below via the VAE's actual temporal_compression, not a
        # hardcoded 4 — different model tiers may differ.
        _condition_indexes_vision = (0, 1)
        try:
            _temporal_compression = int(pipeline.vae.config.scale_factor_temporal)
        except Exception:
            _temporal_compression = 4  # fallback: Wan-family VAEs (used by Cosmos3) default
        _overlap_frames = max(_condition_indexes_vision) * _temporal_compression + 1

        sig = inspect.signature(pipeline.__call__)

        pil_init = None
        if init_image is not None:
            _img_np = (init_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_init = Image.fromarray(_img_np, mode="RGB")

        t_start = time.time()
        all_new_frames = []   # list of Tensor[t, H, W, C] in [0,1], non-overlapping segments
        prev_tail_pil = None  # list[PIL.Image] — tail of previous chunk, fed as `video`
        chunk_idx = 0
        total_frames = 0

        while True:
            elapsed = time.time() - t_start
            if elapsed >= max_seconds:
                print(f"[Cosmos3 Chunked] Stopping: max_seconds budget reached "
                      f"({elapsed:.1f}s / {max_seconds:.0f}s) after {chunk_idx} chunk(s), "
                      f"{total_frames} frames")
                break
            if total_frames >= max_total_frames:
                print(f"[Cosmos3 Chunked] Stopping: max_total_frames reached "
                      f"({total_frames} frames)")
                break

            kwargs = dict(
                prompt=prompt,
                width=w,
                height=h,
                num_frames=chunk_frames,
                fps=float(fps),
                generator=torch.Generator(device=mm.get_torch_device()).manual_seed(seed + chunk_idx),
            )
            if chunk_idx == 0:
                if pil_init is not None:
                    kwargs["image"] = pil_init
            else:
                kwargs["video"] = prev_tail_pil
                kwargs["condition_frame_indexes_vision"] = _condition_indexes_vision
                kwargs["condition_video_keep"] = "first"
            if negative_prompt:
                kwargs["negative_prompt"] = negative_prompt
            if "num_inference_steps" in sig.parameters:
                kwargs["num_inference_steps"] = num_inference_steps
            if "guidance_scale" in sig.parameters:
                kwargs["guidance_scale"] = guidance_scale
            if "output_type" in sig.parameters:
                kwargs["output_type"] = "pt"

            print(f"[Cosmos3 Chunked] chunk {chunk_idx}: {w}×{h} | {chunk_frames} frames | "
                  f"mode={'I2V-start' if (chunk_idx == 0 and pil_init is not None) else ('continuation' if chunk_idx else 'T2V-start')} | "
                  f"elapsed={elapsed:.1f}s/{max_seconds:.0f}s")

            result = pipeline(**kwargs)
            frames_tchw = result[0].float().clamp(0.0, 1.0)   # [T, C, H, W]
            frames_thwc = frames_tchw.permute(0, 2, 3, 1)      # [T, H, W, C]

            if chunk_idx == 0:
                new_segment = frames_thwc
            else:
                # First _overlap_frames reproduce the anchor we fed in — drop them,
                # they'd duplicate the tail already appended from the previous chunk.
                new_segment = frames_thwc[_overlap_frames:]
                if new_segment.shape[0] == 0:
                    print(f"[Cosmos3 Chunked] WARNING: chunk {chunk_idx} produced no new "
                          f"frames beyond the overlap window (chunk_frames={chunk_frames} <= "
                          f"overlap={_overlap_frames}) — stopping to avoid an infinite loop")
                    break

            all_new_frames.append(new_segment)
            total_frames += new_segment.shape[0]

            # Tail for the NEXT chunk's conditioning comes from this chunk's actual
            # output (real generated pixels, not the input anchor).
            _tail = frames_thwc[-_overlap_frames:]
            prev_tail_pil = [
                Image.fromarray((f.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), mode="RGB")
                for f in _tail
            ]

            chunk_idx += 1

        elapsed_total = time.time() - t_start
        full = torch.cat(all_new_frames, dim=0) if all_new_frames else torch.zeros(1, h, w, 3)
        print(f"[Cosmos3 Chunked] Done: {chunk_idx} chunks, {full.shape[0]} total frames "
              f"({full.shape[0] / fps:.1f}s of video @ {fps:.0f}fps), {elapsed_total:.1f}s elapsed")

        # ── save MP4 ──────────────────────────────────────────────────────────
        video_path = ""
        try:
            import imageio.v3 as iio
            frames_np = (full.cpu().numpy() * 255).astype(np.uint8)

            _safe = filename_prefix.strip("/\\")
            _save_dir = os.path.join(
                folder_paths.get_output_directory(),
                os.path.dirname(_safe) or "cosmos3/t2v_long"
            )
            os.makedirs(_save_dir, exist_ok=True)
            _base = os.path.basename(_safe) or "t2v_long"
            _fname = f"{_base}_{seed}_{full.shape[0]}f.mp4"
            video_path = os.path.join(_save_dir, _fname)

            iio.imwrite(video_path, frames_np, fps=fps, codec="h264", quality=8)
            print(f"[Cosmos3 Chunked] Saved {full.shape[0]} frames → {video_path}")
        except Exception as _ve:
            print(f"[Cosmos3 Chunked] Warning: MP4 save failed ({_ve})")

        first_frame = full[0:1]  # [1, H, W, C]
        return (first_frame, video_path, full.shape[0], elapsed_total)


# ── registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "Cosmos3ModelLoader":        Cosmos3ModelLoader,
    "Cosmos3PromptEnricher":     Cosmos3PromptEnricher,
    "Cosmos3T2ISampler":         Cosmos3T2ISampler,
    "Cosmos3T2VSampler":         Cosmos3T2VSampler,
    "Cosmos3T2VChunkedSampler":  Cosmos3T2VChunkedSampler,
    "Cosmos3LoadImageFromURL":   Cosmos3LoadImageFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Cosmos3ModelLoader":        "Cosmos3 Model Loader",
    "Cosmos3PromptEnricher":     "Cosmos3 Prompt Enricher",
    "Cosmos3T2ISampler":         "Cosmos3 T2I / I2I Sampler",
    "Cosmos3T2VSampler":         "Cosmos3 T2V Sampler",
    "Cosmos3T2VChunkedSampler":  "Cosmos3 T2V Chunked (Long-Form) Sampler",
    "Cosmos3LoadImageFromURL":   "Cosmos3 Load Image From URL",
}
