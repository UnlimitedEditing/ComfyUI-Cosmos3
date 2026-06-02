import os
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
        try:
            from diffusers import DiffusionPipeline
        except ImportError:
            raise ImportError(
                "[Cosmos3] diffusers not found. "
                "Install with: pip install git+https://github.com/huggingface/diffusers.git"
            )
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

        pipe = DiffusionPipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            device_map="auto",          # lets accelerate handle CPU offload if VRAM is tight
        )
        print("[Cosmos3] Pipeline loaded.")
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
    Cosmos3-Nano Text-to-Image sampler.

    Prompt: plain text string (wire from Cosmos3PromptEnricher or any STRING node).
    Output: standard ComfyUI IMAGE tensor (1, H, W, C) float32.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":             ("COSMOS3_PIPELINE",),
                "prompt":               ("STRING", {"multiline": True, "forceInput": True}),
                "resolution":           (list(RESOLUTIONS.keys()), {"default": "1280x720 (16:9 HD)"}),
                "num_inference_steps":  ("INT",   {"default": 30,  "min": 1,   "max": 100, "step": 1}),
                "guidance_scale":       ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "seed":                 ("INT",   {"default": 0,   "min": 0,   "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
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
        negative_prompt=None,
        custom_width=1280,
        custom_height=720,
    ):
        if resolution == "custom":
            w, h = custom_width, custom_height
        else:
            w, h = RESOLUTIONS[resolution]

        # Cosmos3 requires dimensions divisible by 16
        w = (w // 16) * 16
        h = (h // 16) * 16

        generator = torch.Generator(device=mm.get_torch_device()).manual_seed(seed)

        kwargs = dict(
            prompt=prompt,
            width=w,
            height=h,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        print(f"[Cosmos3 T2I] Generating {w}x{h} | steps={num_inference_steps} cfg={guidance_scale} seed={seed}")
        result = pipeline(**kwargs)

        image_tensor = pil2tensor(result.images[0])
        return (image_tensor,)


# ── registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "Cosmos3ModelLoader":    Cosmos3ModelLoader,
    "Cosmos3PromptEnricher": Cosmos3PromptEnricher,
    "Cosmos3T2ISampler":     Cosmos3T2ISampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Cosmos3ModelLoader":    "Cosmos3 Model Loader",
    "Cosmos3PromptEnricher": "Cosmos3 Prompt Enricher",
    "Cosmos3T2ISampler":     "Cosmos3 T2I Sampler",
}
