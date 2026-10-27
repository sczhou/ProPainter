"""Official SAM2.1 click segmentation/video propagation with ProPainter removal."""
from contextlib import nullcontext
from pathlib import Path
import gc
import os
import tempfile

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SAM2_MODELS = {
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


def paint_mask(image, mask):
    from tools.painter import mask_painter
    painted = image.copy()
    for object_id in np.unique(mask):
        if object_id > 0:
            painted = mask_painter(painted, (mask == object_id).astype("uint8"),
                                   mask_color=int(object_id) + 1)
    return painted


class SAM2Segmenter:
    def __init__(self, device=None, sam2_model="small", weights_dir=None):
        self.device = torch.device(device or (
            "cuda" if os.environ.get("SPACES_ZERO_GPU") or torch.cuda.is_available() else "cpu"
        ))
        self.sam2_model = sam2_model
        self.weights_dir = Path(weights_dir or os.environ.get("PROPAINTER_WEIGHTS_DIR") or ROOT / "weights")
        self.video_predictor = None
        self.image_predictor = None
        self._inpainter = None

    def select_model(self, name):
        if name not in SAM2_MODELS:
            raise ValueError(f"Unknown SAM2 model: {name}")
        if name != self.sam2_model:
            self.image_predictor = None
            self.video_predictor = None
            self.sam2_model = name
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def precision_context(self):
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    def ensure_checkpoint(self, name=None):
        """Cache one checkpoint on disk without constructing a model or using GPU."""
        name = name or self.sam2_model
        if name not in SAM2_MODELS:
            raise ValueError(f"Unknown SAM2 model: {name}")
        filename = SAM2_MODELS[name][1]
        checkpoint = self.weights_dir / "sam2" / filename
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            return str(checkpoint)
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        repo_id = f"facebook/sam2.1-hiera-{name.replace('_', '-')}"
        # Spaces preloads files into the standard Hub cache during image build.
        # Reuse those files directly, including when HF_HOME was customized.
        for cache_dir in (None, str(Path.home() / ".cache" / "huggingface" / "hub")):
            cached = try_to_load_from_cache(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
            if isinstance(cached, str) and Path(cached).is_file():
                return cached
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename, local_dir=self.weights_dir / "sam2",
        )

    def load_sam2(self):
        if self.video_predictor is not None:
            return
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        config, _ = SAM2_MODELS[self.sam2_model]
        checkpoint = self.ensure_checkpoint()
        self.video_predictor = build_sam2_video_predictor(
            config, checkpoint, device=str(self.device), apply_postprocessing=True,
        )
        # Hole filling uses the optional CUDA extension. Keep all other official
        # stability/conditioning defaults on CPU without requiring that extension.
        if self.device.type != "cuda":
            self.video_predictor.fill_hole_area = 0
        self.image_predictor = SAM2ImagePredictor(self.video_predictor)

    @torch.inference_mode()
    def first_frame_click(self, image, points, labels, *, refine=True):
        if len(points) == 0 or len(points) != len(labels):
            raise ValueError("Provide matching point coordinates and labels.")
        self.load_sam2()
        with self.precision_context():
            # Reset embeddings per call to isolate videos and users.
            self.image_predictor.set_image(image)
            try:
                masks, scores, logits = self.image_predictor.predict(
                    point_coords=np.asarray(points, dtype=np.float32),
                    point_labels=np.asarray(labels, dtype=np.int32),
                    multimask_output=True,
                )
                # MatAnyone's click controller makes a second prediction after
                # a positive click, using the best low-resolution logits as a
                # mask prompt. A single points-only pass misses this refinement.
                if refine and int(labels[-1]) == 1:
                    best = int(np.argmax(scores))
                    masks, scores, logits = self.image_predictor.predict(
                        point_coords=np.asarray(points, dtype=np.float32),
                        point_labels=np.asarray(labels, dtype=np.int32),
                        mask_input=logits[best][None, :, :],
                        multimask_output=True,
                    )
            finally:
                self.image_predictor.reset_predictor()
        best = int(np.argmax(scores))
        return masks[best].astype(np.uint8), logits[best]

    @torch.inference_mode()
    def generator(self, images, template_mask, progress_callback=None):
        if len(images) == 0:
            raise ValueError("The tracking interval is empty.")
        object_ids = [int(i) for i in np.unique(template_mask) if i > 0]
        if not object_ids:
            raise ValueError("Click an object and add a mask before tracking.")
        def report(completed, description):
            if progress_callback is not None:
                progress_callback(completed, len(images), description)

        report(0, "Loading SAM2.1")
        self.load_sam2()
        predictor = self.video_predictor
        # Official SAM2 takes JPEG directories or MP4, unlike MiniMax's fork.
        with tempfile.TemporaryDirectory(prefix="propainter-sam2-") as directory:
            report(0, "Preparing video")
            for i, image in enumerate(images):
                path = str(Path(directory) / f"{i:06d}.jpg")
                if not cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"Could not write tracking frame {i}.")
            with self.precision_context():
                state = predictor.init_state(
                    video_path=directory, offload_video_to_cpu=True,
                    offload_state_to_cpu=self.device.type != "cpu",
                )
                try:
                    for object_id in object_ids:
                        predictor.add_new_mask(
                            inference_state=state, frame_idx=0, obj_id=object_id,
                            mask=template_mask == object_id,
                        )
                    masks = [np.zeros(images[0].shape[:2], np.int32) for _ in images]
                    report(0, "Segmenting video")
                    completed = 0
                    for frame_idx, ids, logits in predictor.propagate_in_video(state):
                        # Resolve overlaps by highest positive logit, retaining IDs.
                        scores = logits[:, 0].float().cpu().numpy()
                        best = scores.argmax(axis=0)
                        ids = np.asarray(ids, dtype=np.int32)
                        masks[frame_idx] = np.where(scores.max(axis=0) > 0, ids[best], 0)
                        completed += 1
                        report(completed, f"Segmenting video — frame {completed}/{len(images)}")
                finally:
                    predictor.reset_state(state)
                    del state
        report(0, "Rendering segmentation preview")
        painted = []
        for i, (image, mask) in enumerate(zip(images, masks), 1):
            painted.append(paint_mask(image, mask))
            report(i, f"Rendering preview — frame {i}/{len(images)}")
        return masks, painted

    @property
    def baseinpainter(self):
        if self._inpainter is None:
            from inpainter.base_inpainter import ProInpainter
            from utils.download_util import load_file_from_url

            base = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"
            checkpoints = [load_file_from_url(base + name, str(self.weights_dir)) for name in (
                "ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth",
            )]
            # torchvision deform_conv2d has no MPS kernel.
            device = "cpu" if self.device.type == "mps" else str(self.device)
            self._inpainter = ProInpainter(*checkpoints, device=torch.device(device),
                                          use_half=device.startswith("cuda"))
        return self._inpainter
