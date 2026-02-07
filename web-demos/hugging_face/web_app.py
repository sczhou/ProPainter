import sys
sys.path.append("../../")

import os
import io
import json
import time
import uuid
import psutil
import argparse

import cv2
import torch
import torchvision
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, session

from tools.painter import mask_painter
from track_anything import TrackingAnything
from model.misc import get_device
from utils.download_util import load_file_from_url


app = Flask(__name__)
app.secret_key = os.urandom(24)

# ---- In-memory session store (single-user / demo use) ----
sessions = {}

# ---- Model loading ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--mask_save', default=False, action='store_true')
    args = parser.parse_args()
    if not args.device:
        args.device = str(get_device())
    return args

args = parse_args()

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
sam_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
checkpoint_folder = os.path.join('..', '..', 'weights')

sam_checkpoint = load_file_from_url(sam_checkpoint_url_dict[args.sam_model_type], checkpoint_folder)
cutie_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'cutie-base-mega.pth'), checkpoint_folder)
propainter_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'ProPainter.pth'), checkpoint_folder)
raft_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'raft-things.pth'), checkpoint_folder)
flow_completion_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), checkpoint_folder)

model = TrackingAnything(sam_checkpoint, cutie_checkpoint, propainter_checkpoint,
                         raft_checkpoint, flow_completion_checkpoint, args)


def get_session(sid):
    """Return session state dict, creating it if needed."""
    if sid not in sessions:
        sessions[sid] = {
            "video_state": {
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
                "mask_save": args.mask_save,
                "multi_mask": {"mask_names": [], "masks": []},
                "track_end_number": None,
            },
            "click_state": [[], []],
        }
    return sessions[sid]


def frame_to_png_bytes(frame_rgb):
    """Convert an RGB numpy frame to PNG bytes."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.png', bgr)
    return buf.tobytes()


def generate_video_from_frames(frames, output_path, fps=30):
    frames_t = torch.from_numpy(np.asarray(frames))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchvision.io.write_video(output_path, frames_t, fps=fps, video_codec="libx264")
    return output_path


# ======== Routes ========

@app.route('/')
def index():
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    return render_template('index.html')


@app.route('/api/session', methods=['POST'])
def create_session():
    sid = str(uuid.uuid4())
    session['sid'] = sid
    get_session(sid)
    return jsonify({"sid": sid})


@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    sid = request.form.get('sid', session.get('sid'))
    if not sid:
        return jsonify({"error": "No session"}), 400

    s = get_session(sid)
    f = request.files.get('video')
    if not f:
        return jsonify({"error": "No video file"}), 400

    os.makedirs('./uploads', exist_ok=True)
    video_path = os.path.join('./uploads', f'{sid}_{f.filename}')
    f.save(video_path)

    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if psutil.virtual_memory().percent > 90:
                break
        cap.release()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not frames:
        return jsonify({"error": "Could not extract frames"}), 400

    h, w = frames[0].shape[:2]

    s["video_state"] = {
        "video_name": f.filename,
        "origin_images": frames,
        "painted_images": [fr.copy() for fr in frames],
        "masks": [np.zeros((h, w), np.uint8) for _ in frames],
        "logits": [None] * len(frames),
        "select_frame_number": 0,
        "fps": fps,
    }
    s["interactive_state"] = {
        "inference_times": 0,
        "negative_click_times": 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {"mask_names": [], "masks": []},
        "track_end_number": None,
    }
    s["click_state"] = [[], []]

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(frames[0])

    return jsonify({
        "total_frames": len(frames),
        "fps": round(fps, 2),
        "width": w,
        "height": h,
        "video_name": f.filename,
    })


@app.route('/api/frame/<int:frame_number>')
def get_frame(frame_number):
    sid = request.args.get('sid', session.get('sid'))
    s = get_session(sid)
    vs = s["video_state"]

    if vs["origin_images"] is None:
        return jsonify({"error": "No video loaded"}), 400

    idx = max(0, min(frame_number, len(vs["origin_images"]) - 1))
    img = vs["painted_images"][idx]
    return send_file(io.BytesIO(frame_to_png_bytes(img)), mimetype='image/png')


@app.route('/api/select-frame', methods=['POST'])
def select_frame():
    data = request.json
    sid = data.get('sid', session.get('sid'))
    frame_number = int(data.get('frame_number', 0))

    s = get_session(sid)
    vs = s["video_state"]
    if vs["origin_images"] is None:
        return jsonify({"error": "No video loaded"}), 400

    idx = max(0, min(frame_number, len(vs["origin_images"]) - 1))
    vs["select_frame_number"] = idx

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(vs["origin_images"][idx])
    s["click_state"] = [[], []]

    return jsonify({"selected": idx})


@app.route('/api/set-end-frame', methods=['POST'])
def set_end_frame():
    data = request.json
    sid = data.get('sid', session.get('sid'))
    frame_number = int(data.get('frame_number', 0))
    s = get_session(sid)
    s["interactive_state"]["track_end_number"] = frame_number
    return jsonify({"end_frame": frame_number})


@app.route('/api/sam-click', methods=['POST'])
def sam_click():
    data = request.json
    sid = data.get('sid', session.get('sid'))
    x = int(data['x'])
    y = int(data['y'])
    label = int(data.get('label', 1))  # 1 = positive, 0 = negative

    s = get_session(sid)
    vs = s["video_state"]
    click_state = s["click_state"]

    if vs["origin_images"] is None:
        return jsonify({"error": "No video loaded"}), 400

    click_state[0].append([x, y])
    click_state[1].append(label)

    frame_idx = vs["select_frame_number"]
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(vs["origin_images"][frame_idx])

    mask, logit, painted_image = model.first_frame_click(
        image=vs["origin_images"][frame_idx],
        points=np.array(click_state[0]),
        labels=np.array(click_state[1]),
        multimask="True",
    )

    vs["masks"][frame_idx] = mask
    vs["logits"][frame_idx] = logit
    vs["painted_images"][frame_idx] = painted_image

    if label == 1:
        s["interactive_state"]["positive_click_times"] += 1
    else:
        s["interactive_state"]["negative_click_times"] += 1

    return jsonify({"status": "ok"})


@app.route('/api/clear-clicks', methods=['POST'])
def clear_clicks():
    data = request.json
    sid = data.get('sid', session.get('sid'))
    s = get_session(sid)
    vs = s["video_state"]
    s["click_state"] = [[], []]
    if vs["origin_images"] is not None:
        idx = vs["select_frame_number"]
        vs["painted_images"][idx] = vs["origin_images"][idx].copy()
    return jsonify({"status": "ok"})


@app.route('/api/add-mask', methods=['POST'])
def add_mask():
    data = request.json
    sid = data.get('sid', session.get('sid'))
    s = get_session(sid)
    vs = s["video_state"]
    ist = s["interactive_state"]

    if vs["origin_images"] is None:
        return jsonify({"error": "No video loaded"}), 400

    mask = vs["masks"][vs["select_frame_number"]]
    ist["multi_mask"]["masks"].append(mask.copy())
    name = "mask_{:03d}".format(len(ist["multi_mask"]["masks"]))
    ist["multi_mask"]["mask_names"].append(name)

    # Repaint with all masks shown
    select_frame = vs["origin_images"][vs["select_frame_number"]].copy()
    for i, m in enumerate(ist["multi_mask"]["masks"]):
        select_frame = mask_painter(select_frame, m.astype('uint8'), mask_color=i + 2)
    vs["painted_images"][vs["select_frame_number"]] = select_frame

    s["click_state"] = [[], []]

    return jsonify({
        "mask_name": name,
        "masks": ist["multi_mask"]["mask_names"],
    })


@app.route('/api/remove-masks', methods=['POST'])
def remove_masks():
    data = request.json
    sid = data.get('sid', session.get('sid'))
    s = get_session(sid)
    ist = s["interactive_state"]
    vs = s["video_state"]
    ist["multi_mask"]["mask_names"] = []
    ist["multi_mask"]["masks"] = []
    if vs["origin_images"] is not None:
        idx = vs["select_frame_number"]
        vs["painted_images"][idx] = vs["origin_images"][idx].copy()
    return jsonify({"status": "ok", "masks": []})


@app.route('/api/track', methods=['POST'])
def track():
    data = request.json
    sid = data.get('sid', session.get('sid'))
    mask_selection = data.get('mask_selection', [])

    s = get_session(sid)
    vs = s["video_state"]
    ist = s["interactive_state"]

    if vs["origin_images"] is None:
        return jsonify({"error": "No video loaded"}), 400

    model.cutie.clear_memory()

    start = vs["select_frame_number"]
    end = ist["track_end_number"]
    if end:
        following_frames = vs["origin_images"][start:end]
    else:
        following_frames = vs["origin_images"][start:]

    # Build combined template mask
    if ist["multi_mask"]["masks"]:
        if not mask_selection:
            mask_selection = ["mask_001"]
        mask_selection.sort()
        first_idx = int(mask_selection[0].split("_")[1]) - 1
        template_mask = ist["multi_mask"]["masks"][first_idx] * (first_idx + 1)
        for i in range(1, len(mask_selection)):
            mi = int(mask_selection[i].split("_")[1]) - 1
            template_mask = np.clip(
                template_mask + ist["multi_mask"]["masks"][mi] * (mi + 1),
                0, mi + 1
            )
        vs["masks"][start] = template_mask
    else:
        template_mask = vs["masks"][start]

    if len(np.unique(template_mask)) == 1:
        return jsonify({"error": "No mask defined. Add at least one mask first."}), 400

    masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask)
    model.cutie.clear_memory()

    if end:
        vs["masks"][start:end] = masks
        vs["logits"][start:end] = logits
        vs["painted_images"][start:end] = painted_images
    else:
        vs["masks"][start:] = masks
        vs["logits"][start:] = logits
        vs["painted_images"][start:] = painted_images

    video_output = generate_video_from_frames(
        vs["painted_images"],
        f"./result/track/{sid}_{vs['video_name']}",
        fps=vs["fps"]
    )
    ist["inference_times"] += 1

    if ist["mask_save"]:
        mask_dir = f'./result/mask/{vs["video_name"].split(".")[0]}'
        os.makedirs(mask_dir, exist_ok=True)
        for i, m in enumerate(vs["masks"]):
            np.save(os.path.join(mask_dir, f'{i:05d}.npy'), m)

    return jsonify({"status": "ok", "video": f"/api/result-video/track?sid={sid}"})


@app.route('/api/inpaint', methods=['POST'])
def inpaint():
    data = request.json
    sid = data.get('sid', session.get('sid'))
    mask_selection = data.get('mask_selection', [])

    s = get_session(sid)
    vs = s["video_state"]

    if vs["origin_images"] is None:
        return jsonify({"error": "No video loaded"}), 400

    resize_ratio = float(data.get('resize_ratio', 1.0))
    dilate_radius = int(data.get('dilate_radius', 8))
    raft_iter = int(data.get('raft_iter', 20))
    subvideo_length = int(data.get('subvideo_length', 80))
    neighbor_length = int(data.get('neighbor_length', 10))
    ref_stride = int(data.get('ref_stride', 10))

    frames = np.asarray(vs["origin_images"])
    inpaint_masks = np.asarray(vs["masks"]).copy()

    if not mask_selection:
        mask_selection = ["mask_001"]
    mask_selection.sort()

    inpaint_mask_numbers = [int(ms.split("_")[1]) for ms in mask_selection]
    unique_masks = np.unique(inpaint_masks)
    for i in range(1, len(unique_masks)):
        val = unique_masks[i]
        if val not in inpaint_mask_numbers:
            inpaint_masks[inpaint_masks == val] = 0

    inpainted_frames = model.baseinpainter.inpaint(
        frames, inpaint_masks,
        ratio=resize_ratio,
        dilate_radius=dilate_radius,
        raft_iter=raft_iter,
        subvideo_length=subvideo_length,
        neighbor_length=neighbor_length,
        ref_stride=ref_stride,
    )

    video_output = generate_video_from_frames(
        inpainted_frames,
        f"./result/inpaint/{sid}_{vs['video_name']}",
        fps=vs["fps"]
    )

    return jsonify({"status": "ok", "video": f"/api/result-video/inpaint?sid={sid}"})


@app.route('/api/result-video/<video_type>')
def result_video(video_type):
    sid = request.args.get('sid', session.get('sid'))
    s = get_session(sid)
    vs = s["video_state"]
    path = f"./result/{video_type}/{sid}_{vs['video_name']}"
    if not os.path.exists(path):
        return jsonify({"error": "Video not found"}), 404
    return send_file(path, mimetype='video/mp4')


@app.route('/api/reset', methods=['POST'])
def reset():
    data = request.json
    sid = data.get('sid', session.get('sid'))
    if sid in sessions:
        del sessions[sid]
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    os.makedirs('./result/track', exist_ok=True)
    os.makedirs('./result/inpaint', exist_ok=True)
    os.makedirs('./uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=args.port, debug=True)
