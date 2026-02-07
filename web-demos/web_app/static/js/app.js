// ── State ──────────────────────────────────────────────────────────────
const state = {
    totalFrames: 0,
    currentFrame: 0,
    endFrame: 0,
    isPositive: true,
    masks: [],
    imageWidth: 0,
    imageHeight: 0,
    canvasScale: 1,
    busy: false,
};

// ── DOM Elements ──────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const uploadArea = $("#upload-area");
const uploadPrompt = $("#upload-prompt");
const videoInput = $("#video-input");
const videoInfo = $("#video-info");
const canvas = $("#frame-canvas");
const ctx = canvas.getContext("2d");
const canvasOverlay = $("#canvas-overlay");

const startSlider = $("#start-frame-slider");
const endSlider = $("#end-frame-slider");
const startLabel = $("#start-frame-label");
const endLabel = $("#end-frame-label");

const btnPositive = $("#btn-positive");
const btnNegative = $("#btn-negative");
const btnAddMask = $("#btn-add-mask");
const btnClearClicks = $("#btn-clear-clicks");
const btnRemoveMasks = $("#btn-remove-masks");
const btnTrack = $("#btn-track");
const btnInpaint = $("#btn-inpaint");
const btnReset = $("#btn-reset");

const maskList = $("#mask-list");
const statusMessage = $("#status-message");
const spinner = $("#spinner");

const step2 = $("#step2");
const step3 = $("#step3");

// Parameter elements
const paramSliders = {
    resize: { slider: $("#param-resize"), display: $("#val-resize") },
    dilate: { slider: $("#param-dilate"), display: $("#val-dilate") },
    raft: { slider: $("#param-raft"), display: $("#val-raft") },
    subvideo: { slider: $("#param-subvideo"), display: $("#val-subvideo") },
    neighbor: { slider: $("#param-neighbor"), display: $("#val-neighbor") },
    stride: { slider: $("#param-stride"), display: $("#val-stride") },
};

// ── Helpers ────────────────────────────────────────────────────────────
function setStatus(msg, type = "") {
    statusMessage.textContent = msg;
    statusMessage.className = "status-message" + (type ? ` ${type}` : "");
}

function showSpinner(show) {
    spinner.style.display = show ? "block" : "none";
}

function showLoading(msg) {
    const overlay = document.createElement("div");
    overlay.className = "loading-overlay";
    overlay.id = "loading-overlay";
    overlay.innerHTML = `<div class="spinner-lg"></div><p>${msg}</p>`;
    document.body.appendChild(overlay);
}

function hideLoading() {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) overlay.remove();
}

function enableStep(step) {
    step.classList.remove("disabled");
}

function disableStep(step) {
    step.classList.add("disabled");
}

function enableControls(enable) {
    const controls = [btnPositive, btnNegative, btnAddMask, btnClearClicks, btnRemoveMasks, startSlider, endSlider];
    controls.forEach((el) => (el.disabled = !enable));
}

async function apiCall(url, options = {}) {
    const res = await fetch(url, {
        credentials: "same-origin",
        ...options,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "Request failed" }));
        throw new Error(err.error || "Request failed");
    }
    return res.json();
}

function drawImageOnCanvas(base64) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            state.imageWidth = img.width;
            state.imageHeight = img.height;

            // Scale canvas to fit container while maintaining aspect ratio
            const container = canvas.parentElement;
            const containerW = container.clientWidth;
            const containerH = container.clientHeight;
            const scale = Math.min(containerW / img.width, containerH / img.height, 1);

            canvas.width = img.width;
            canvas.height = img.height;
            canvas.style.width = `${img.width * scale}px`;
            canvas.style.height = `${img.height * scale}px`;
            state.canvasScale = scale;

            ctx.drawImage(img, 0, 0);
            resolve();
        };
        img.src = "data:image/jpeg;base64," + base64;
    });
}

function updateMaskList() {
    if (state.masks.length === 0) {
        maskList.innerHTML = '<span class="empty-state">No masks added</span>';
    } else {
        maskList.innerHTML = state.masks
            .map((name, i) => `<span class="mask-tag color-${i % 6}">${name}</span>`)
            .join("");
    }
}

// ── Upload ─────────────────────────────────────────────────────────────
uploadArea.addEventListener("click", () => videoInput.click());
uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
});
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("dragover"));
uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});
videoInput.addEventListener("change", () => {
    if (videoInput.files.length) handleFile(videoInput.files[0]);
});

async function handleFile(file) {
    if (state.busy) return;
    state.busy = true;
    showSpinner(true);
    setStatus("Uploading and extracting frames...");

    const formData = new FormData();
    formData.append("video", file);

    try {
        const data = await apiCall("/api/upload", {
            method: "POST",
            body: formData,
        });

        state.totalFrames = data.total_frames;
        state.currentFrame = 0;
        state.endFrame = data.total_frames - 1;
        state.masks = [];

        // Show video info
        $("#info-name").textContent = data.video_name;
        $("#info-fps").textContent = data.fps;
        $("#info-frames").textContent = data.total_frames;
        $("#info-size").textContent = `${data.width} x ${data.height}`;
        videoInfo.style.display = "block";
        uploadArea.style.display = "none";

        // Setup sliders
        startSlider.max = data.total_frames - 1;
        startSlider.value = 0;
        startLabel.textContent = "0";
        endSlider.max = data.total_frames - 1;
        endSlider.value = data.total_frames - 1;
        endLabel.textContent = String(data.total_frames - 1);

        // Draw first frame
        canvasOverlay.classList.add("hidden");
        await drawImageOnCanvas(data.first_frame);

        // Enable steps
        enableStep(step2);
        enableStep(step3);
        enableControls(true);
        updateMaskList();

        setStatus("Video loaded. Click on the image to select objects for removal.", "success");
    } catch (err) {
        setStatus("Upload failed: " + err.message, "error");
    } finally {
        state.busy = false;
        showSpinner(false);
    }
}

// ── Reset ──────────────────────────────────────────────────────────────
btnReset.addEventListener("click", async () => {
    await apiCall("/api/reset", { method: "POST" });
    location.reload();
});

// ── Frame Selection ───────────────────────────────────────────────────
startSlider.addEventListener("input", () => {
    startLabel.textContent = startSlider.value;
});

startSlider.addEventListener("change", async () => {
    if (state.busy) return;
    state.busy = true;
    showSpinner(true);
    const frame = parseInt(startSlider.value);
    state.currentFrame = frame;
    setStatus(`Selecting frame ${frame}...`);

    try {
        const data = await apiCall("/api/select-frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame }),
        });
        await drawImageOnCanvas(data.frame);
        setStatus(`Frame ${frame} selected. Click on objects to create masks.`, "success");
    } catch (err) {
        setStatus("Error: " + err.message, "error");
    } finally {
        state.busy = false;
        showSpinner(false);
    }
});

endSlider.addEventListener("input", () => {
    endLabel.textContent = endSlider.value;
});

endSlider.addEventListener("change", async () => {
    if (state.busy) return;
    const endFrame = parseInt(endSlider.value);
    state.endFrame = endFrame;
    try {
        await apiCall("/api/set-end-frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame: endFrame }),
        });
        setStatus(`Tracking end frame set to ${endFrame}.`);
    } catch (err) {
        setStatus("Error: " + err.message, "error");
    }
});

// ── Click Mode Toggle ─────────────────────────────────────────────────
btnPositive.addEventListener("click", () => {
    state.isPositive = true;
    btnPositive.classList.add("active");
    btnNegative.classList.remove("active");
});

btnNegative.addEventListener("click", () => {
    state.isPositive = false;
    btnNegative.classList.add("active");
    btnPositive.classList.remove("active");
});

// ── Canvas Click (SAM) ────────────────────────────────────────────────
canvas.addEventListener("click", async (e) => {
    if (state.busy) return;
    state.busy = true;
    showSpinner(true);

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    setStatus(`Processing click at (${x}, ${y})...`);

    try {
        const data = await apiCall("/api/sam-click", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ x, y, positive: state.isPositive }),
        });
        await drawImageOnCanvas(data.frame);
        setStatus('Mask generated. Click "Add Mask" to keep it, or click again to refine.', "success");
    } catch (err) {
        setStatus("Error: " + err.message, "error");
    } finally {
        state.busy = false;
        showSpinner(false);
    }
});

// ── Mask Actions ──────────────────────────────────────────────────────
btnAddMask.addEventListener("click", async () => {
    if (state.busy) return;
    state.busy = true;
    showSpinner(true);
    setStatus("Adding mask...");

    try {
        const data = await apiCall("/api/add-mask", { method: "POST" });
        state.masks = data.masks;
        await drawImageOnCanvas(data.frame);
        updateMaskList();
        btnTrack.disabled = false;
        setStatus(`Mask added (${state.masks.length} total). Add more or proceed to tracking.`, "success");
    } catch (err) {
        setStatus("Error: " + err.message, "error");
    } finally {
        state.busy = false;
        showSpinner(false);
    }
});

btnClearClicks.addEventListener("click", async () => {
    if (state.busy) return;
    state.busy = true;
    showSpinner(true);

    try {
        const data = await apiCall("/api/clear-clicks", { method: "POST" });
        await drawImageOnCanvas(data.frame);
        setStatus("Clicks cleared. Click on the image to start a new mask.");
    } catch (err) {
        setStatus("Error: " + err.message, "error");
    } finally {
        state.busy = false;
        showSpinner(false);
    }
});

btnRemoveMasks.addEventListener("click", async () => {
    if (state.busy) return;
    state.busy = true;
    showSpinner(true);

    try {
        const data = await apiCall("/api/remove-masks", { method: "POST" });
        state.masks = [];
        await drawImageOnCanvas(data.frame);
        updateMaskList();
        btnTrack.disabled = true;
        btnInpaint.disabled = true;
        setStatus("All masks removed.");
    } catch (err) {
        setStatus("Error: " + err.message, "error");
    } finally {
        state.busy = false;
        showSpinner(false);
    }
});

// ── Tracking ──────────────────────────────────────────────────────────
btnTrack.addEventListener("click", async () => {
    if (state.busy) return;
    state.busy = true;
    showLoading("Tracking objects across frames... This may take a while.");
    setStatus("Running tracking...");

    try {
        const data = await apiCall("/api/track", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ masks: state.masks }),
        });

        // Show tracking result
        const trackResult = $("#track-result");
        const trackVideo = $("#track-video");
        trackVideo.src = data.video_url + "?t=" + Date.now();
        trackResult.style.display = "block";
        btnInpaint.disabled = false;
        setStatus("Tracking complete! You can now run inpainting.", "success");
    } catch (err) {
        setStatus("Tracking failed: " + err.message, "error");
    } finally {
        state.busy = false;
        hideLoading();
    }
});

// ── Inpainting ────────────────────────────────────────────────────────
btnInpaint.addEventListener("click", async () => {
    if (state.busy) return;
    state.busy = true;
    showLoading("Running ProPainter inpainting... This may take a while.");
    setStatus("Running inpainting...");

    const params = {
        masks: state.masks,
        resize_ratio: parseFloat(paramSliders.resize.slider.value),
        dilate_radius: parseInt(paramSliders.dilate.slider.value),
        raft_iter: parseInt(paramSliders.raft.slider.value),
        subvideo_length: parseInt(paramSliders.subvideo.slider.value),
        neighbor_length: parseInt(paramSliders.neighbor.slider.value),
        ref_stride: parseInt(paramSliders.stride.slider.value),
    };

    try {
        const data = await apiCall("/api/inpaint", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        });

        const inpaintResult = $("#inpaint-result");
        const inpaintVideo = $("#inpaint-video");
        inpaintVideo.src = data.video_url + "?t=" + Date.now();
        inpaintResult.style.display = "block";
        setStatus("Inpainting complete! The result video is ready.", "success");
    } catch (err) {
        setStatus("Inpainting failed: " + err.message, "error");
    } finally {
        state.busy = false;
        hideLoading();
    }
});

// ── Parameter sliders live update ─────────────────────────────────────
Object.values(paramSliders).forEach(({ slider, display }) => {
    slider.addEventListener("input", () => {
        const val = parseFloat(slider.value);
        display.textContent = Number.isInteger(val) ? val : val.toFixed(2);
    });
});
