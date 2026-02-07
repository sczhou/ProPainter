import sys
sys.path.append("../../")
sys.path.append("../hugging_face/")

import os
import io
import json
import time
import uuid
import psutil
import base64
import argparse
import tempfile

import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file, session

from tools.painter import mask_painter, point_painter
from track_anything import TrackingAnything
from model.misc import get_device
from utils.download_util import load_file_from_url

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ── Global state per session (simple single-user version) ──────────────────
sessions = {}


def get_session():
    sid = session.get("sid")
    if sid and sid in sessions:
        return sessions[sid]
    return None


def create_session():
    sid = str(uuid.uuid4())
    session["sid"] = sid
    sessions[sid] = {
        "video_state": {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30,
        },
        "interactive_state": {
            "inference_times": 0,
            "negative_click_times": 0,
            "positive_click_times": 0,
            "mask_save": False,
            "multi_mask": {"mask_names": [], "masks": []},
            "track_end_number": None,
        },
        "click_state": [[], []],
    }
    return sessions[sid]


# ── Model initialisation ──────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mask_save", default=False, action="store_true")
    args = parser.parse_args()
    if not args.device:
        args.device = str(get_device())
    return args


args = parse_args()

pretrain_model_url = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"
sam_checkpoint_url_dict = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
checkpoint_folder = os.path.join("..", "..", "weights")

sam_checkpoint = load_file_from_url(sam_checkpoint_url_dict[args.sam_model_type], checkpoint_folder)
cutie_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, "cutie-base-mega.pth"), checkpoint_folder)
propainter_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, "ProPainter.pth"), checkpoint_folder)
raft_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, "raft-things.pth"), checkpoint_folder)
flow_completion_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, "recurrent_flow_completion.pth"), checkpoint_folder)

model = TrackingAnything(sam_checkpoint, cutie_checkpoint, propainter_checkpoint, raft_checkpoint, flow_completion_checkpoint, args)


# ── Helpers ────────────────────────────────────────────────────────────────
def numpy_to_base64(img_array):
    """Convert a numpy RGB image to a base64 JPEG string."""
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    img_bgr = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def generate_video_from_frames(frames, output_path, fps=30):
    frames_tensor = torch.from_numpy(np.asarray(frames))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchvision.io.write_video(output_path, frames_tensor, fps=fps, video_codec="libx264")
    return output_path


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Upload a video, extract frames, initialise session state."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    sess = create_session()
    video_file = request.files["video"]

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.save(tmp.name)
    tmp.close()

    frames = []
    try:
        cap = cv2.VideoCapture(tmp.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if psutil.virtual_memory().percent > 90:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp.name)

    if len(frames) == 0:
        return jsonify({"error": "Could not extract any frames from the video"}), 400

    h, w = frames[0].shape[:2]
    sess["video_state"] = {
        "user_name": time.time(),
        "video_name": video_file.filename or "video.mp4",
        "origin_images": frames,
        "painted_images": [f.copy() for f in frames],
        "masks": [np.zeros((h, w), np.uint8) for _ in range(len(frames))],
        "logits": [None] * len(frames),
        "select_frame_number": 0,
        "fps": fps,
    }

    # Set first frame for SAM
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(frames[0])

    return jsonify({
        "total_frames": len(frames),
        "fps": round(fps, 2),
        "width": w,
        "height": h,
        "video_name": video_file.filename,
        "first_frame": numpy_to_base64(frames[0]),
    })


@app.route("/api/frame/<int:frame_num>")
def get_frame(frame_num):
    """Return a specific frame as base64 JPEG."""
    sess = get_session()
    if not sess:
        return jsonify({"error": "No session"}), 400
    vs = sess["video_state"]
    if vs["origin_images"] is None:
        return jsonify({"error": "No video loaded"}), 400
    idx = max(0, min(frame_num, len(vs["origin_images"]) - 1))
    return jsonify({"frame": numpy_to_base64(vs["painted_images"][idx]), "index": idx})


@app.route("/api/select-frame", methods=["POST"])
def select_frame():
    """Select a template frame for mask painting."""
    sess = get_session()
    if not sess:
        return jsonify({"error": "No session"}), 400

    data = request.json
    frame_idx = data.get("frame", 0)
    vs = sess["video_state"]
    frame_idx = max(0, min(frame_idx, len(vs["origin_images"]) - 1))
    vs["select_frame_number"] = frame_idx

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(vs["origin_images"][frame_idx])

    # Reset click state for new frame
    sess["click_state"] = [[], []]

    return jsonify({
        "frame": numpy_to_base64(vs["painted_images"][frame_idx]),
        "index": frame_idx,
    })


@app.route("/api/set-end-frame", methods=["POST"])
def set_end_frame():
    """Set the tracking end frame."""
    sess = get_session()
    if not sess:
        return jsonify({"error": "No session"}), 400

    data = request.json
    end_frame = data.get("frame")
    sess["interactive_state"]["track_end_number"] = end_frame
    return jsonify({"end_frame": end_frame})


@app.route("/api/sam-click", methods=["POST"])
def sam_click():
    """Process a click on the frame for SAM segmentation."""
    sess = get_session()
    if not sess:
        return jsonify({"error": "No session"}), 400

    data = request.json
    x = data["x"]
    y = data["y"]
    is_positive = data.get("positive", True)

    vs = sess["video_state"]
    click_state = sess["click_state"]

    click_state[0].append([x, y])
    click_state[1].append(1 if is_positive else 0)

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(vs["origin_images"][vs["select_frame_number"]])

    prompt = {
        "prompt_type": ["click"],
        "input_point": click_state[0],
        "input_label": click_state[1],
        "multimask_output": "True",
    }

    mask, logit, painted_image = model.first_frame_click(
        image=vs["origin_images"][vs["select_frame_number"]],
        points=np.array(prompt["input_point"]),
        labels=np.array(prompt["input_label"]),
        multimask=prompt["multimask_output"],
    )

    vs["masks"][vs["select_frame_number"]] = mask
    vs["logits"][vs["select_frame_number"]] = logit
    vs["painted_images"][vs["select_frame_number"]] = np.array(painted_image)

    return jsonify({"frame": numpy_to_base64(painted_image)})


@app.route("/api/add-mask", methods=["POST"])
def add_mask():
    """Add current mask to the multi-mask set."""
    sess = get_session()
    if not sess:
        return jsonify({"error": "No session"}), 400

    vs = sess["video_state"]
    ist = sess["interactive_state"]

    mask = vs["masks"][vs["select_frame_number"]]
    mask_name = "mask_{:03d}".format(len(ist["multi_mask"]["masks"]) + 1)
    ist["multi_mask"]["masks"].append(mask.copy())
    ist["multi_mask"]["mask_names"].append(mask_name)

    # Repaint frame with all masks
    select_frame = vs["origin_images"][vs["select_frame_number"]].copy()
    for i, m in enumerate(ist["multi_mask"]["masks"]):
        select_frame = mask_painter(select_frame, m.astype("uint8"), mask_color=i + 2)

    vs["painted_images"][vs["select_frame_number"]] = select_frame

    # Reset click state
    sess["click_state"] = [[], []]

    return jsonify({
        "frame": numpy_to_base64(select_frame),
        "masks": ist["multi_mask"]["mask_names"],
    })


@app.route("/api/remove-masks", methods=["POST"])
def remove_masks():
    """Remove all masks."""
    sess = get_session()
    if not sess:
        return jsonify({"error": "No session"}), 400

    ist = sess["interactive_state"]
    vs = sess["video_state"]
    ist["multi_mask"]["masks"] = []
    ist["multi_mask"]["mask_names"] = []

    # Restore original frame
    frame_idx = vs["select_frame_number"]
    vs["painted_images"][frame_idx] = vs["origin_images"][frame_idx].copy()
    vs["masks"][frame_idx] = np.zeros_like(vs["masks"][frame_idx])

    return jsonify({
        "frame": numpy_to_base64(vs["origin_images"][frame_idx]),
        "masks": [],
    })


@app.route("/api/clear-clicks", methods=["POST"])
def clear_clicks():
    """Clear current click state and restore frame."""
    sess = get_session()
    if not sess:
        return jsonify({"error": "No session"}), 400

    sess["click_state"] = [[], []]
    vs = sess["video_state"]
    frame_idx = vs["select_frame_number"]

    # Repaint with existing multi-masks only
    ist = sess["interactive_state"]
    select_frame = vs["origin_images"][frame_idx].copy()
    for i, m in enumerate(ist["multi_mask"]["masks"]):
        select_frame = mask_painter(select_frame, m.astype("uint8"), mask_color=i + 2)

    vs["painted_images"][frame_idx] = select_frame

    return jsonify({"frame": numpy_to_base64(select_frame)})


@app.route("/api/track", methods=["POST"])
def track_video():
    """Run VOS tracking on the video."""
    sess = get_session()
    if not sess:
        return jsonify({"error": "No session"}), 400

    vs = sess["video_state"]
    ist = sess["interactive_state"]

    mask_selection = request.json.get("masks", ist["multi_mask"]["mask_names"])
    if not mask_selection:
        mask_selection = ["mask_001"] if ist["multi_mask"]["masks"] else []

    if not mask_selection:
        return jsonify({"error": "No masks to track. Please add at least one mask."}), 400

    model.cutie.clear_memory()

    if ist["track_end_number"]:
        following_frames = vs["origin_images"][vs["select_frame_number"]:ist["track_end_number"]]
    else:
        following_frames = vs["origin_images"][vs["select_frame_number"]:]

    # Build template mask
    mask_selection_sorted = sorted(mask_selection)
    template_mask = ist["multi_mask"]["masks"][int(mask_selection_sorted[0].split("_")[1]) - 1] * int(mask_selection_sorted[0].split("_")[1])
    for i in range(1, len(mask_selection_sorted)):
        mask_number = int(mask_selection_sorted[i].split("_")[1]) - 1
        template_mask = np.clip(
            template_mask + ist["multi_mask"]["masks"][mask_number] * (mask_number + 1),
            0, mask_number + 1,
        )

    vs["masks"][vs["select_frame_number"]] = template_mask

    if len(np.unique(template_mask)) == 1:
        template_mask[0][0] = 1

    masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask)
    model.cutie.clear_memory()

    if ist["track_end_number"]:
        vs["masks"][vs["select_frame_number"]:ist["track_end_number"]] = masks
        vs["logits"][vs["select_frame_number"]:ist["track_end_number"]] = logits
        vs["painted_images"][vs["select_frame_number"]:ist["track_end_number"]] = painted_images
    else:
        vs["masks"][vs["select_frame_number"]:] = masks
        vs["logits"][vs["select_frame_number"]:] = logits
        vs["painted_images"][vs["select_frame_number"]:] = painted_images

    # Generate tracking output video
    output_dir = os.path.join(".", "result", "track")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, vs["video_name"])
    painted_np = [np.array(img) if isinstance(img, Image.Image) else img for img in vs["painted_images"]]
    video_path = generate_video_from_frames(painted_np, output_path, fps=vs["fps"])

    sess["track_video_path"] = os.path.abspath(video_path)
    ist["inference_times"] += 1

    return jsonify({"status": "ok", "video_url": "/api/result/track"})


@app.route("/api/inpaint", methods=["POST"])
def inpaint_video():
    """Run ProPainter inpainting."""
    sess = get_session()
    if not sess:
        return jsonify({"error": "No session"}), 400

    vs = sess["video_state"]
    ist = sess["interactive_state"]
    data = request.json or {}

    resize_ratio = data.get("resize_ratio", 1.0)
    dilate_radius = data.get("dilate_radius", 8)
    raft_iter = data.get("raft_iter", 20)
    subvideo_length = data.get("subvideo_length", 80)
    neighbor_length = data.get("neighbor_length", 10)
    ref_stride = data.get("ref_stride", 10)
    mask_selection = data.get("masks", ist["multi_mask"]["mask_names"])

    if not mask_selection:
        mask_selection = ["mask_001"]

    frames = np.asarray(vs["origin_images"])
    inpaint_masks = np.asarray(vs["masks"])

    mask_selection_sorted = sorted(mask_selection)
    inpaint_mask_numbers = [int(m.split("_")[1]) for m in mask_selection_sorted]
    unique_masks = np.unique(inpaint_masks)
    num_masks = len(unique_masks) - 1
    for i in range(1, num_masks + 1):
        if i in inpaint_mask_numbers:
            continue
        inpaint_masks[inpaint_masks == i] = 0

    inpainted_frames = model.baseinpainter.inpaint(
        frames, inpaint_masks,
        ratio=resize_ratio,
        dilate_radius=dilate_radius,
        raft_iter=raft_iter,
        subvideo_length=subvideo_length,
        neighbor_length=neighbor_length,
        ref_stride=ref_stride,
    )

    output_dir = os.path.join(".", "result", "inpaint")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, vs["video_name"])
    video_path = generate_video_from_frames(inpainted_frames, output_path, fps=vs["fps"])
    sess["inpaint_video_path"] = os.path.abspath(video_path)

    return jsonify({"status": "ok", "video_url": "/api/result/inpaint"})


@app.route("/api/result/<result_type>")
def get_result(result_type):
    """Stream the result video."""
    sess = get_session()
    if not sess:
        return jsonify({"error": "No session"}), 400

    key = f"{result_type}_video_path"
    path = sess.get(key)
    if not path or not os.path.exists(path):
        return jsonify({"error": "No result available"}), 404

    return send_file(path, mimetype="video/mp4", as_attachment=False)


@app.route("/api/reset", methods=["POST"])
def reset_session():
    """Reset the current session."""
    sid = session.get("sid")
    if sid and sid in sessions:
        del sessions[sid]
    return jsonify({"status": "ok"})


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=True)
