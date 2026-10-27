import sys
import os
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import time
import argparse

def startup_log(message):
    print(f"[Startup] {message}", flush=True)


startup_log("Starting ProPainter demo")
# HF's build-time preloader uses the standard Hub cache. Keep that cache on
# Spaces; use a repository-local cache only for local runs.
if not os.environ.get("SPACE_ID"):
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# Keep HF ZeroGPU support without requiring the Spaces SDK on local machines.
if os.environ.get("SPACES_ZERO_GPU"):
    startup_log("Initializing ZeroGPU SDK")
    import spaces
else:
    def _local_gpu(fn=None, **kwargs):
        return fn if fn is not None else lambda callback: callback
    spaces = SimpleNamespace(GPU=_local_gpu)
startup_log("Importing OpenCV, PyTorch and Gradio")
import cv2
import torch
import numpy as np
import gradio as gr
import av

from tools.painter import mask_painter, point_painter
from sam2_segmenter import SAM2Segmenter, SAM2_MODELS

from model.misc import get_device
from utils.download_util import download_url_to_file

startup_log("Dependencies imported")


def download_example(url, path):
    startup_log(f"Downloading example {Path(path).name}")
    download_url_to_file(url, path)
    startup_log(f"Example ready: {Path(path).name}")

# make sample videos into mp4 as git does not allow mp4 without lfs
sample_videos_path = str(HERE / "test_sample")
os.makedirs(sample_videos_path, exist_ok=True)
if not os.path.exists(os.path.join(sample_videos_path, "test-sample0.mp4")):
    download_example("https://github-production-user-asset-6210df.s3.amazonaws.com/14334509/281805130-e57c7016-5a6d-4d3b-9df9-b4ea6372cc87.mp4", os.path.join(sample_videos_path, "test-sample0.mp4"))
if not os.path.exists(os.path.join(sample_videos_path, "test-sample1.mp4")):
    download_example("https://github-production-user-asset-6210df.s3.amazonaws.com/14334509/281828039-5def0fc9-3a22-45b7-838d-6bf78b6772c3.mp4", os.path.join(sample_videos_path, "test-sample1.mp4"))
if not os.path.exists(os.path.join(sample_videos_path, "test-sample2.mp4")):
    download_example("https://github-production-user-asset-6210df.s3.amazonaws.com/76810782/281807801-69b9f70c-1e56-428d-9b1b-4870c5e533a7.mp4", os.path.join(sample_videos_path, "test-sample2.mp4"))
if not os.path.exists(os.path.join(sample_videos_path, "test-sample3.mp4")):
    download_example("https://github-production-user-asset-6210df.s3.amazonaws.com/76810782/281808625-ad98f03f-99c7-4008-acf1-3d7beb48f13b.mp4", os.path.join(sample_videos_path, "test-sample3.mp4"))
if not os.path.exists(os.path.join(sample_videos_path, "test-sample4.mp4")):
    download_example("https://github-production-user-asset-6210df.s3.amazonaws.com/14334509/281828066-ee09ae82-916f-4a2e-a6c7-6fc50645fd20.mp4", os.path.join(sample_videos_path, "test-sample4.mp4"))


def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument("--sam2-model", choices=SAM2_MODELS, default="small")
    parser.add_argument('--port', type=int, default=7860, help="only useful when running gradio applications")
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])
    
    if not args.device:
        args.device = str(get_device())

    parser_host = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    args.host = parser_host
    return args 

MAX_VIDEO_FRAMES = 500


def new_video_state():
    return {"user_name": "", "video_name": "", "origin_images": [],
            "painted_images": [], "masks": [], "logits": [],
            "select_frame_number": 0, "fps": 30., "pending_mask": None,
            "pending_frame": None, "sam2_model": args.sam2_model}


def new_interactive_state():
    return {"inference_times": 0, "negative_click_times": 0,
            "positive_click_times": 0, "mask_save": args.mask_save,
            "multi_mask": {"mask_names": [], "masks": []}, "track_end_number": None}


def require_video(video_state):
    if not video_state.get("origin_images"):
        raise gr.Error("Please load a video first.")


def get_frames_from_video(video_input, video_state, sam2_model=None):
    if not video_input:
        raise gr.Error("Please upload a video first.")
    cap = cv2.VideoCapture(video_input)
    frames = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) > MAX_VIDEO_FRAMES:
            raise gr.Error("The uploaded video contains more than 500 frames. Please upload a shorter clip.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if len(frames) >= MAX_VIDEO_FRAMES:
                raise gr.Error("The uploaded video contains more than 500 frames. Please upload a shorter clip.")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise gr.Error("Failed to read any frames from the video. Please try uploading again.")
    # Use MatAnyone's upload resizing policy before assigning the template frame.
    h, w = frames[0].shape[:2]
    if h >= 1080 and w >= 1080:
        scale = 1080 / min(h, w)
        frames = [cv2.resize(frame, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA) for frame in frames]
    video_state = new_video_state()
    video_state["sam2_model"] = sam2_model or args.sam2_model
    video_state.update(user_name=time.time(), video_name=Path(video_input).name,
                       origin_images=frames, painted_images=frames.copy(), fps=fps,
                       masks=[np.zeros(frames[0].shape[:2], np.int32) for _ in frames],
                       logits=[None] * len(frames))
    return (video_state, new_interactive_state(), [[], []],
            gr.update(value=frames[0], visible=True),
            gr.update(visible=True, maximum=max(2, len(frames)), value=1),
            gr.update(visible=False, maximum=max(2, len(frames)), value=len(frames)),
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
            gr.update(visible=False), gr.update(visible=True),
            gr.update(visible=True, choices=[], value=[]),
            gr.update(visible=True, value=None), gr.update(visible=True, value=None),
            gr.update(visible=True), gr.update(visible=True),
            gr.update(value=None, visible=True, interactive=False))


def _invalidate_tracking(video_state):
    video_state.pop("track_range", None)
    frames = video_state.get("origin_images")
    if frames:
        video_state["masks"] = [np.zeros(frames[0].shape[:2], np.int32) for _ in frames]
        video_state["logits"] = [None] * len(frames)
        video_state["painted_images"] = frames.copy()


def select_template(number, video_state, interactive_state):
    require_video(video_state)
    index = max(0, min(int(number) - 1, len(video_state["origin_images"]) - 1))
    video_state["select_frame_number"] = index
    video_state["pending_mask"], video_state["pending_frame"] = None, None
    interactive_state["multi_mask"] = {"mask_names": [], "masks": []}
    _invalidate_tracking(video_state)
    return (video_state["origin_images"][index], video_state, interactive_state,
            [[], []], gr.update(choices=[], value=[]), None, None, gr.update(value=None, interactive=False))



def change_model(choice, video_state, interactive_state):
    video_state["sam2_model"] = choice
    video_state["pending_mask"], video_state["pending_frame"] = None, None
    interactive_state["multi_mask"] = {"mask_names": [], "masks": []}
    _invalidate_tracking(video_state)
    frames = video_state.get("origin_images")
    frame = frames[video_state["select_frame_number"]].copy() if frames else None
    return (video_state, interactive_state, [[], []], gr.update(choices=[], value=[]),
            frame, None, None, gr.update(value=None, interactive=False))


def get_end_number(number, interactive_state):
    interactive_state["track_end_number"] = int(number)
    return interactive_state


@spaces.GPU(duration=60)
def _sam_refine_gpu(frame, click_state, click_index, is_positive, sam2_model="small"):
    # MatAnyone returns click history explicitly across the GPU worker boundary.
    points = [list(point) for point in click_state[0]] + [list(map(int, click_index))]
    labels = list(click_state[1]) + [int(is_positive)]
    model.select_model(sam2_model)
    mask, logit = model.first_frame_click(frame, np.asarray(points), np.asarray(labels))
    painted = mask_painter(frame, mask.astype("uint8"), 3, 0.7, 2, 5)
    coords, categories = np.asarray(points), np.asarray(labels)
    painted = point_painter(painted, coords[categories > 0], 8, 0.9, 15, 2, 5)
    painted = point_painter(painted, coords[categories < 1], 50, 0.9, 15, 2, 5)
    return mask, logit, painted, [points, labels]


def sam_refine(video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData, progress=gr.Progress()):
    require_video(video_state)
    index = video_state["select_frame_number"]
    positive = point_prompt == "Positive"
    variant = video_state.get("sam2_model", args.sam2_model)
    progress(0, desc=f"Preparing SAM2.1 {variant.replace('_', ' ')} checkpoint")
    # Download on CPU before entering the ZeroGPU worker. Selecting a variant
    # alone never downloads it or allocates model memory.
    model.ensure_checkpoint(variant)
    progress(0, desc="Generating mask")
    mask, logit, painted, click_state = _sam_refine_gpu(
        video_state["origin_images"][index], click_state, evt.index, positive,
        variant)
    video_state["pending_mask"], video_state["pending_frame"] = mask, index
    video_state["painted_images"][index] = painted
    interactive_state["positive_click_times" if positive else "negative_click_times"] += 1
    return painted, video_state, interactive_state, click_state


def show_mask(video_state, interactive_state, mask_dropdown):
    frames = video_state.get("origin_images")
    if not frames:
        return None
    painted = frames[video_state["select_frame_number"]].copy()
    multi = interactive_state["multi_mask"]
    for i, (name, mask) in enumerate(zip(multi["mask_names"], multi["masks"])):
        if name in (mask_dropdown or []):
            painted = mask_painter(painted, mask.astype("uint8"), mask_color=i + 2)
    return painted


def add_multi_mask(video_state, interactive_state, mask_dropdown):
    require_video(video_state)
    mask = video_state.get("pending_mask")
    if (mask is None or video_state.get("pending_frame") != video_state["select_frame_number"]
            or not mask.any()):
        raise gr.Error("Please click the image to generate a mask first.")
    multi = interactive_state["multi_mask"]
    multi["masks"].append(mask.copy())
    name = "mask_{:03d}".format(len(multi["masks"]))
    multi["mask_names"].append(name)
    selected = list(mask_dropdown or []) + [name]
    video_state["pending_mask"], video_state["pending_frame"] = None, None
    _invalidate_tracking(video_state)
    painted = show_mask(video_state, interactive_state, selected)
    video_state["painted_images"][video_state["select_frame_number"]] = painted
    return (interactive_state, gr.update(choices=list(multi["mask_names"]), value=selected),
            painted, [[], []], video_state, None, None, gr.update(value=None, interactive=False))


def clear_click(video_state, click_state):
    require_video(video_state)
    video_state["pending_mask"], video_state["pending_frame"] = None, None
    frame = video_state["origin_images"][video_state["select_frame_number"]].copy()
    video_state["painted_images"][video_state["select_frame_number"]] = frame
    return frame, [[], []], video_state


def remove_multi_mask(video_state, interactive_state, mask_dropdown):
    interactive_state["multi_mask"] = {"mask_names": [], "masks": []}
    video_state["pending_mask"], video_state["pending_frame"] = None, None
    _invalidate_tracking(video_state)
    frames = video_state.get("origin_images")
    frame = frames[video_state["select_frame_number"]].copy() if frames else None
    return video_state, interactive_state, gr.update(choices=[], value=[]), frame, [[], []], None, None, gr.update(value=None, interactive=False)


@spaces.GPU(duration=300)
def segment_video(video_state, interactive_state, mask_dropdown, progress=gr.Progress()):
    require_video(video_state)
    frames = video_state["origin_images"]
    start = video_state["select_frame_number"]
    end = min(interactive_state["track_end_number"] or len(frames), len(frames))
    if end <= start:
        raise gr.Error("The tracking interval is empty.")
    template = np.zeros(frames[0].shape[:2], np.int32)
    multi = interactive_state["multi_mask"]
    if multi["masks"]:
        selected = mask_dropdown or [multi["mask_names"][0]]
        for i, (name, mask) in enumerate(zip(multi["mask_names"], multi["masks"])):
            if name in selected:
                template[mask > 0] = i + 1
    elif video_state.get("pending_frame") == start and video_state.get("pending_mask") is not None:
        template[video_state["pending_mask"] > 0] = 1
    if not template.any():
        raise gr.Error("Please select an object before running Video Segmentation.")
    model.select_model(video_state.get("sam2_model", args.sam2_model))
    report = lambda done, total, description: progress(done / total, desc=description)
    masks, painted = model.generator(frames[start:end], template, progress_callback=report)
    _invalidate_tracking(video_state)
    video_state["masks"][start:end] = masks
    video_state["painted_images"][start:end] = painted
    video_state["track_range"] = (start, end)
    interactive_state["inference_times"] += 1
    if interactive_state["mask_save"]:
        folder = ROOT / "results" / "sam2-demo" / str(video_state["user_name"]) / "masks"
        folder.mkdir(parents=True, exist_ok=True)
        for i, mask in enumerate(video_state["masks"]):
            np.save(folder / f"{i:05d}.npy", mask)
    output = generate_video_from_frames(video_state["painted_images"], _result_path(video_state, "segmentation"), video_state["fps"],
                                        progress_callback=report)
    return output, video_state, interactive_state


def clear_inpainting_result():
    return None, gr.update(value=None, interactive=False)


@spaces.GPU(duration=300)
def inpaint_video(video_state, ratio, dilation, raft_iters, subvideo, neighbors, stride, mask_dropdown, progress=gr.Progress()):
    require_video(video_state)
    if "track_range" not in video_state:
        raise gr.Error("Run Video Segmentation before Video Inpainting.")
    start, end = video_state["track_range"]
    if end - start < 2:
        raise gr.Error("Video Inpainting needs at least two segmented frames.")
    ids = [int(name.split("_")[1]) for name in (mask_dropdown or ["mask_001"])]
    masks = np.isin(np.asarray(video_state["masks"]), ids).astype(np.uint8)
    if not masks.any():
        raise gr.Error("Please segment the selected objects first.")
    frames = video_state["origin_images"]
    progress(0, desc="Loading ProPainter")
    report = lambda done, total, description: progress(done / total, desc=description)
    result = model.baseinpainter.inpaint(
        np.asarray(frames[start:end]), masks[start:end], ratio=float(ratio),
        dilate_radius=int(dilation), raft_iter=int(raft_iters), subvideo_length=int(subvideo),
        neighbor_length=int(neighbors), ref_stride=int(stride), progress_callback=report)
    h, w = result[0].shape[:2]
    output_frames = [cv2.resize(frame, (w, h)) for frame in frames]
    output_frames[start:end] = result
    output = generate_video_from_frames(output_frames, _result_path(video_state, "inpainting"), video_state["fps"],
                                        progress_callback=report)
    return output, gr.update(value=output, interactive=True)


def _result_path(video_state, kind):
    return str(ROOT / "results" / "sam2-demo" / str(video_state["user_name"]) / kind /
               Path(video_state["video_name"]).with_suffix(".mp4").name)


def generate_video_from_frames(frames, output_path, fps=30, progress_callback=None):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    with av.open(output_path, "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(float(fps)).limit_denominator(10000))
        stream.width, stream.height, stream.pix_fmt = w + w % 2, h + h % 2, "yuv420p"
        if progress_callback is not None:
            progress_callback(0, len(frames), "Encoding video")
        for i, image in enumerate(frames, 1):
            image = np.pad(np.asarray(image, np.uint8), ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")):
                container.mux(packet)
            if progress_callback is not None:
                progress_callback(i, len(frames), f"Encoding video — frame {i}/{len(frames)}")
        for packet in stream.encode():
            container.mux(packet)
    return output_path


def restart():
    return (new_video_state(), new_interactive_state(), [[], []],
            gr.update(value=None, visible=False),
            gr.update(value=1, visible=False), gr.update(value=1, visible=False),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(choices=[], value=[], visible=False),
            gr.update(value=None, visible=False), gr.update(value=None, visible=False),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(value=None, visible=False, interactive=False))


args = parse_augment()
startup_log(f"Selected device: {args.device}; default SAM2.1 variant: {args.sam2_model}")
model = SAM2Segmenter(device=args.device, sam2_model=args.sam2_model)
if model.device.type == "cpu":
    torch.set_num_threads(min(8, os.cpu_count() or 1))


title = r"""<h1 style="text-align: center; font-size: 2em; font-weight: 700; line-height: 1.25;"><span style="color: #fb5a42;">ProPainter</span>: Improving <span style="color: #fb5a42;">Propa</span>gat<span style="color: #fb5a42;">i</span>o<span style="color: #fb5a42;">n</span> and <span style="color: #fb5a42;">T</span>ransform<span style="color: #fb5a42;">er</span> for Video Inpainting</h1>"""

description = r"""
<center><img src='https://github.com/sczhou/ProPainter/raw/main/assets/propainter_logo1_glow.png' alt='Propainter logo' style="width:180px; margin-bottom:20px"></center>
<b>Official Gradio demo</b> for <a href='https://github.com/sczhou/ProPainter' target='_blank'><b>Improving Propagation and Transformer for Video Inpainting (ICCV 2023)</b></a>.<br>
🔥 Propainter is a robust inpainting algorithm.<br>
🤗 Try to drop your video, add the masks and get the the inpainting results!<br>
"""
article = r"""
<div><span>If ProPainter is helpful, please help to ⭐ the <a href="https://github.com/sczhou/ProPainter" target="_blank">Github Repo</a>. Thanks!</span><a href="https://github.com/sczhou/ProPainter"><img src="https://img.shields.io/github/stars/sczhou/ProPainter?style=social" alt="GitHub Stars"></a></div>

---

📝 **Citation**
<br>
If our work is useful for your research, please consider citing:
```bibtex
@inproceedings{zhou2023propainter,
   title={{ProPainter}: Improving Propagation and Transformer for Video Inpainting},
   author={Zhou, Shangchen and Li, Chongyi and Chan, Kelvin C.K and Loy, Chen Change},
   booktitle={ICCV},
   year={2023}
}
```

📋 **License**
<br>
This project is licensed under <a rel="license" href="https://github.com/sczhou/CodeFormer/blob/master/LICENSE">S-Lab License 1.0</a>. 
Redistribution and use for non-commercial purposes should follow this license.

📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>shangchenzhou@gmail.com</b>.
<div class="find-me"><span>🤗 Find Me:</span><a href="https://twitter.com/ShangchenZhou"><img src="https://img.shields.io/twitter/follow/ShangchenZhou?label=%40ShangchenZhou&style=social" alt="Twitter Follow"></a><a href="https://github.com/sczhou"><img src="https://img.shields.io/github/followers/sczhou?style=social" alt="Github Follow"></a></div>

"""
css = r"""
.gradio-container {width: 85% !important; margin: 0 auto !important; --segmentation-color: #4CAF50; --inpainting-color: #fb5a42;}
#model-settings {border: 1px solid var(--border-color-primary) !important; border-radius: 5px !important; box-shadow: none !important;}
button {border-radius: 8px !important;}
.new_button {background-color: var(--button-secondary-background-fill) !important; color: var(--button-secondary-text-color) !important; border: 1px solid var(--border-color-primary) !important;}
.green_button {background-color: #4CAF50 !important; color: #ffffff !important; border: none !important;}
.orange_button {background-color: rgb(243 111 91) !important; color: #ffffff !important; border: none !important;}
.new_button:hover {background-color: var(--button-secondary-background-fill-hover) !important;}
.green_button:hover {background-color: #77bd79 !important;}
.orange_button:hover {background-color: #fc7b69 !important;}

/* Keep real Gradio progress flush with each output video's lower edge. */
#segmentation-output {--loader-color: var(--segmentation-color);}
#inpainting-output {--loader-color: var(--inpainting-color);}
.output-video .wrap.default:has(.progress-text) {
    top: auto !important; bottom: 0 !important; left: 0 !important; right: 0 !important;
    height: 28px !important; padding: 0 !important; border-radius: 0 !important;
    background: var(--block-background-fill) !important; pointer-events: none;
}
.output-video .wrap.default:has(.progress-text) svg,
.output-video .eta-bar {display: none !important;}
.output-video .progress-level {position: absolute; bottom: 0; left: 0; right: 0; align-items: stretch;}
.output-video .progress-level-inner {
    width: calc(100% - 144px); max-width: calc(100% - 144px); margin: 0 136px 6px 8px !important;
    font-family: inherit !important; font-size: 12px !important; line-height: 16px !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-align: left;
}
.output-video .progress-text.meta-text {
    right: 8px !important; bottom: 10px !important; padding: 0 !important;
    font-family: inherit !important; font-size: 12px !important; line-height: 16px !important;
    text-align: right; white-space: nowrap; background: transparent !important;
}
.output-video .progress-bar-wrap {
    width: 100% !important; height: 4px !important; border: 0 !important;
    background: var(--border-color-primary) !important;
}
#segmentation-output .progress-bar {background-color: #4CAF50 !important;}
#inpainting-output .progress-bar {background-color: #fb5a42 !important;}

.mask_button_group {gap: 10px !important; flex-grow: 0 !important;}
.step2-heading {align-items: center !important;}
.step-heading-column {justify-content: center !important;}
/* The template must upscale like the video, even for low-resolution uploads. */
#template_frame .image-frame img {object-fit: contain !important;}
.margin_center {width: 50% !important; margin: auto !important;}
.jc_center {justify-content: center !important;}
.video-title {
    margin-bottom: 5px !important;
}
.custom-bg {
    background-color: var(--background-fill-secondary);
    padding: 10px;
    border-radius: 10px;
}
.title-container {
    text-align: center;
    padding: 0;
    margin: 0;
    height: 4vh;
    width: 80vw;
    font-family: "Sarpanch", sans-serif;
    font-weight: 60;
}
#custom-markdown {
    font-family: "Roboto", sans-serif;
    font-size: 18px;
    color: #333333;
    font-weight: bold;
}
small {
    font-size: 60%;
}

/* Gradio sanitizes inner HTML classes in Markdown; use the component scope. */
.project-info div:has(> a > img) {display: flex; align-items: center; gap: 8px; flex-wrap: nowrap; margin-top: 0.5em; margin-bottom: 1em;}
.project-info div:has(> a > img) > a {display: inline-flex; align-items: center; flex: 0 0 auto;}
.project-info div:has(> a > img) img {display: block; margin: 0 !important;}
.project-info div:has(> a[href="https://twitter.com/ShangchenZhou"]) > span {white-space: nowrap;}
"""

with gr.Blocks() as iface:
    click_state = gr.State([[], []])
    interactive_state = gr.State(new_interactive_state())
    video_state = gr.State(new_video_state())
    gr.HTML(title)
    gr.Markdown(description)

    with gr.Accordion("Model Settings (click to expand)", open=False, elem_id="model-settings"):
        with gr.Row():
            resize_ratio_number = gr.Slider(.01, 1, value=1, step=.01, label="Resize ratio")
            raft_iter_number = gr.Slider(5, 20, value=20, step=1, label="Iterations for RAFT inference.")
        with gr.Row():
            dilate_radius_number = gr.Slider(0, 10, value=8, step=1, label="Mask dilation for video and flow masking.")
            subvideo_length_number = gr.Slider(40, 200, value=80, step=1, label="Length of sub-video for long video inference.")
        with gr.Row():
            neighbor_length_number = gr.Slider(5, 20, value=10, step=1, label="Length of local neighboring frames.")
            ref_stride_number = gr.Slider(5, 20, value=10, step=1, label="Stride of global reference frames.")
        with gr.Row():
            image_selection_slider = gr.Slider(1, 100, value=1, step=1, label="Start Frame", visible=False)
            track_pause_number_slider = gr.Slider(1, 100, value=100, step=1, label="Track end frame", visible=False)
        with gr.Row():
            point_prompt = gr.Radio(["Positive", "Negative"], value="Positive", label="Point Prompt", interactive=True, visible=False, min_width=100, scale=1)
            mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask Selection", visible=False)

    gr.Markdown("---")
    with gr.Column():
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, elem_classes="step-heading-column"):
                gr.Markdown("## Step1: Upload video")
            with gr.Column(scale=2, elem_classes="step-heading-column"):
                with gr.Row(visible=False, elem_classes="step2-heading") as step2_title:
                    with gr.Column(scale=3, min_width=260):
                        gr.Markdown("## Step2: Add masks <small>(Several clicks then **`Add Mask`** <u>one by one</u>)</small>")
                    model_selection = gr.Dropdown(
                        choices=[("SAM2.1 Small", "small"),
                                 ("SAM2.1 Base+", "base_plus"), ("SAM2.1 Large", "large")],
                        value=args.sam2_model, label="SAM2.1 version", show_label=False,
                        container=False, interactive=True, filterable=False,
                        scale=0, min_width=160)
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                video_input = gr.Video(label="Input Video", elem_classes="video")
                extract_frames_button = gr.Button("Load Video", interactive=True, elem_classes="new_button")
            with gr.Column(scale=2):
                template_frame = gr.Image(label="Start Frame", type="pil", interactive=True, elem_id="template_frame", visible=False, elem_classes="image")
                with gr.Row(equal_height=True, elem_classes="mask_button_group"):
                    clear_button_click = gr.Button("Clear Clicks", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                    add_mask_button = gr.Button("Add Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                    remove_mask_button = gr.Button("Remove Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                    segmentation_button = gr.Button("Step 3: Video Segmentation", interactive=True, visible=False, elem_classes="green_button", min_width=100)

        gr.HTML('<hr style="border: none; height: 1.5px; background: linear-gradient(to right, #4CAF50, rgb(243 111 91));margin: 5px 0;">')
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                segmentation_output = gr.Video(label="Segmentation Output", visible=False,
                                               elem_id="segmentation-output", elem_classes=["video", "output-video"])
            with gr.Column(scale=2):
                inpainting_output = gr.Video(label="Inpainting Output", visible=False,
                                             elem_id="inpainting-output", elem_classes=["video", "output-video"])
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                inpainting_button = gr.Button("Step 4: Video Inpainting", interactive=True, visible=False, elem_classes="orange_button", min_width=100)
            with gr.Column(scale=2):
                download_button = gr.DownloadButton("Download", interactive=False, visible=False, elem_classes="new_button", min_width=100)

    load_outputs = [video_state, interactive_state, click_state, template_frame,
                    image_selection_slider, track_pause_number_slider, point_prompt,
                    clear_button_click, add_mask_button, remove_mask_button, segmentation_button,
                    mask_dropdown, segmentation_output, inpainting_output, inpainting_button, step2_title, download_button]
    extract_frames_button.click(get_frames_from_video, [video_input, video_state, model_selection], load_outputs, concurrency_id="models")
    image_selection_slider.release(select_template, [image_selection_slider, video_state, interactive_state],
                                   [template_frame, video_state, interactive_state, click_state, mask_dropdown, segmentation_output, inpainting_output, download_button], concurrency_id="models")
    track_pause_number_slider.release(get_end_number, [track_pause_number_slider, interactive_state], [interactive_state], concurrency_id="models")
    template_frame.select(sam_refine, [video_state, point_prompt, click_state, interactive_state],
                          [template_frame, video_state, interactive_state, click_state], concurrency_id="models",
                          show_progress_on=template_frame)
    add_mask_button.click(add_multi_mask, [video_state, interactive_state, mask_dropdown],
                          [interactive_state, mask_dropdown, template_frame, click_state, video_state, segmentation_output, inpainting_output, download_button], concurrency_id="models")
    clear_button_click.click(clear_click, [video_state, click_state], [template_frame, click_state, video_state], concurrency_id="models")
    remove_mask_button.click(remove_multi_mask, [video_state, interactive_state, mask_dropdown],
                             [video_state, interactive_state, mask_dropdown, template_frame, click_state, segmentation_output, inpainting_output, download_button], concurrency_id="models")
    mask_dropdown.input(show_mask, [video_state, interactive_state, mask_dropdown], [template_frame], concurrency_id="models")
    segmentation_button.click(clear_inpainting_result, outputs=[inpainting_output, download_button],
                               concurrency_id="models", show_progress="hidden", api_name=False).then(
        segment_video, [video_state, interactive_state, mask_dropdown],
        [segmentation_output, video_state, interactive_state],
        concurrency_id="models", show_progress_on=segmentation_output)
    inpainting_button.click(inpaint_video,
                            [video_state, resize_ratio_number, dilate_radius_number, raft_iter_number,
                             subvideo_length_number, neighbor_length_number, ref_stride_number, mask_dropdown],
                            [inpainting_output, download_button], concurrency_id="models",
                            show_progress_on=inpainting_output)
    model_selection.input(change_model, [model_selection, video_state, interactive_state],
                          [video_state, interactive_state, click_state, mask_dropdown, template_frame,
                           segmentation_output, inpainting_output, download_button], concurrency_id="models")
    video_input.clear(restart, outputs=load_outputs, concurrency_id="models", show_progress=False)
    gr.Markdown("---")
    gr.Markdown("## Examples")
    gr.Examples(examples=[str(HERE / "test_sample" / f"test-sample{i}.mp4") for i in range(5)], inputs=[video_input])
    gr.Markdown(article, elem_classes="project-info")

iface.queue()
if __name__ == "__main__":
    # Cache every offered checkpoint before serving. Model construction remains
    # lazy, with only the selected variant loaded for inference.
    for variant in SAM2_MODELS:
        startup_log(f"Preparing SAM2.1 {variant} checkpoint")
        path = model.ensure_checkpoint(variant)
        startup_log(f"SAM2.1 {variant} checkpoint ready: {path}")
    startup_log(f"Launching Gradio on {args.host}:{args.port}")
    iface.launch(debug=True, server_name=args.host, server_port=args.port,
                 theme=gr.themes.Monochrome(), css=css,
                 allowed_paths=[str(ROOT / "results"), str(HERE / "test_sample")])
