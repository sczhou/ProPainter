## Get Started

From the repository root (Python 3.10+, tested with Python 3.11 on macOS arm64):

```bash
python3 -m venv .venv
source .venv/bin/activate
SAM2_BUILD_CUDA=0 python -m pip install -r web-demos/hugging_face/requirements.txt
python web-demos/hugging_face/app.py
```

Open http://127.0.0.1:7860. On CUDA machines, use a matching PyTorch build and
optionally omit `SAM2_BUILD_CUDA=0`.

## Usage

The UI follows MatAnyone's 2×2 video layout and button styles:

1. Upload a video at the upper left and click **Load Video**.
2. Click objects in the upper-right frame. **Add Mask** saves the current mask
   and starts fresh clicks for the next object. **Clear Clicks** clears only the
   draft. Reloading the video or changing its start frame clears the added masks.
3. Click **Step 3: Video Segmentation** to get the lower-left result.
4. Click **Step 4: Video Inpainting** below the segmentation result to get the lower-right
   result. Use **Download** below that result to save the generated video.

Positive/negative prompts, mask selection, and ProPainter processing parameters
are in **Model Settings**. Changing the start frame clears the old frame's masks.
Videos exceeding 500 frames are rejected with a popup when loading; no static
limit notice or Video Info box is displayed.

Segmentation uses only official SAM2.1. The legacy SAM1/Cutie code and dependencies
are removed. ProPainter performs inpainting, including RAFT and flow completion.
The default SAM2 model is **SAM2.1 Small**. The dropdown beside **Step2** offers
Small, Base+, and Large; both click segmentation and video propagation use the
selected checkpoint. Switching models clears old prompts/masks and outputs.
Positive clicks use the same two-pass logits refinement as MatAnyone's click
controller. MatAnyone's default click model is SAM1 ViT-H; this demo keeps SAM2
and does not load SAM1 or a matting model. Use `--device cuda` on a CUDA machine.
At startup all three checkpoints (Small, Base+, and Large) are downloaded to disk
before the demo starts serving. No SAM2 network is constructed at startup.
On Hugging Face Spaces, `preload_from_hub` downloads these checkpoints during the
image build instead; startup reads the standard Hub cache without downloading
them again. Startup stages are written immediately to the container logs.
Changing the dropdown only selects a variant; only that variant is loaded into
memory on the first click/inference. No download progress bar is added to the
dropdown. Missing checkpoint recovery downloads run before the ZeroGPU worker
starts. Cached checkpoint files are reused directly, without another download or
remote metadata request.
Only one SAM2 variant is kept loaded; the previous variant is released when the
next variant is used. ProPainter/RAFT/flow-completion weights remain lazy until
the first inpainting request.

Weights are cached in `weights/sam2/` and `weights/`. Set
`PROPAINTER_WEIGHTS_DIR` to a writable persistent directory to relocate all model
weights on a cloud deployment. Ephemeral container storage does not preserve the
cache across container replacement/rebuild; use persistent storage or a mounted
Hugging Face Storage Bucket if that is required. The original example videos
download if absent.
Outputs are saved under `results/sam2-demo/` (gitignored) and contain no audio.
CPU inference is slow for full videos. CUDA/MPS and HF deployment are untested.

## Citation
If you find our repo useful for your research, please consider citing our paper:
```bibtex
@inproceedings{zhou2023propainter,
   title={{ProPainter}: Improving Propagation and Transformer for Video Inpainting},
   author={Zhou, Shangchen and Li, Chongyi and Chan, Kelvin C.K and Loy, Chen Change},
   booktitle={ICCV},
   year={2023}
}
```


## License

This project is licensed under <a rel="license" href="../../LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.


## Acknowledgements

The segmentation demo uses [SAM2](https://github.com/facebookresearch/sam2), with ProPainter for video inpainting.
