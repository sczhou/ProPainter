// ======== State ========
const state = {
    sid: null,
    totalFrames: 0,
    currentStep: 1,
    clickMode: 'positive', // 'positive' or 'negative'
    masks: [],             // list of mask names from server
    videoInfo: null,
    trackingDone: false,
    imageNaturalWidth: 0,
    imageNaturalHeight: 0,
};

// ======== DOM refs ========
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// Step 1
const uploadZone = $('#upload-zone');
const videoFileInput = $('#video-file');
const videoPreview = $('#video-preview');
const videoPreviewArea = $('#video-preview-area');
const uploadLoading = $('#upload-loading');

// Step 2
const maskCanvas = $('#mask-canvas');
const ctx = maskCanvas.getContext('2d');
const startFrameSlider = $('#start-frame-slider');
const endFrameSlider = $('#end-frame-slider');
const startFrameValue = $('#start-frame-value');
const endFrameValue = $('#end-frame-value');
const canvasLoading = $('#canvas-loading');
const maskList = $('#mask-list');

// Step 3
const settingSliders = {
    resize: { input: $('#setting-resize'), display: $('#val-resize') },
    raftIter: { input: $('#setting-raft-iter'), display: $('#val-raft-iter') },
    dilate: { input: $('#setting-dilate'), display: $('#val-dilate') },
    subvideo: { input: $('#setting-subvideo'), display: $('#val-subvideo') },
    neighbor: { input: $('#setting-neighbor'), display: $('#val-neighbor') },
    refStride: { input: $('#setting-ref-stride'), display: $('#val-ref-stride') },
};

// ======== Init ========
document.addEventListener('DOMContentLoaded', init);

async function init() {
    // Create session
    const res = await fetch('/api/session', { method: 'POST' });
    const data = await res.json();
    state.sid = data.sid;

    setupUpload();
    setupStep2();
    setupStep3();
    setupSettings();
    setupNavigation();
}

// ======== Notifications ========
function notify(msg, type = 'info') {
    const el = $('#notification');
    el.textContent = msg;
    el.className = `notification ${type}`;
    el.classList.remove('hidden');
    if (type !== 'error') {
        setTimeout(() => el.classList.add('hidden'), 4000);
    }
}

function clearNotification() {
    $('#notification').classList.add('hidden');
}

// ======== Step Navigation ========
function goToStep(step) {
    state.currentStep = step;
    // Update stepper UI
    $$('.step').forEach((el) => {
        const s = parseInt(el.dataset.step);
        el.classList.remove('active', 'completed');
        if (s === step) el.classList.add('active');
        else if (s < step) el.classList.add('completed');
    });
    // Show/hide panels
    $$('.panel').forEach((p) => p.classList.remove('active-panel'));
    $(`#step${step}-panel`).classList.add('active-panel');

    clearNotification();

    if (step === 2) {
        loadFrame(parseInt(startFrameSlider.value));
    }
    if (step === 3) {
        populateStep3Masks();
    }
}

function setupNavigation() {
    $('#btn-to-step2').addEventListener('click', () => goToStep(2));
    $('#btn-to-step3').addEventListener('click', () => {
        if (state.masks.length === 0) {
            notify('Add at least one mask before proceeding.', 'error');
            return;
        }
        goToStep(3);
    });
    $('#btn-back-step1').addEventListener('click', () => goToStep(1));
    $('#btn-back-step2').addEventListener('click', () => goToStep(2));
    $('#btn-restart').addEventListener('click', async () => {
        await fetch('/api/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sid: state.sid }),
        });
        location.reload();
    });
}

// ======== Step 1: Upload ========
function setupUpload() {
    uploadZone.addEventListener('click', () => videoFileInput.click());

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            videoFileInput.files = e.dataTransfer.files;
            handleVideoUpload(e.dataTransfer.files[0]);
        }
    });

    videoFileInput.addEventListener('change', () => {
        if (videoFileInput.files.length > 0) {
            handleVideoUpload(videoFileInput.files[0]);
        }
    });

    $('#btn-change-video').addEventListener('click', () => {
        videoPreviewArea.classList.add('hidden');
        uploadZone.classList.remove('hidden');
        videoFileInput.value = '';
    });
}

async function handleVideoUpload(file) {
    uploadZone.classList.add('hidden');
    uploadLoading.classList.remove('hidden');

    const formData = new FormData();
    formData.append('video', file);
    formData.append('sid', state.sid);

    try {
        const res = await fetch('/api/upload-video', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Upload failed');

        state.totalFrames = data.total_frames;
        state.videoInfo = data;

        // Show preview
        videoPreview.src = URL.createObjectURL(file);
        $('#info-name').textContent = data.video_name;
        $('#info-resolution').textContent = `${data.width} x ${data.height}`;
        $('#info-fps').textContent = data.fps;
        $('#info-frames').textContent = data.total_frames;

        // Update sliders
        startFrameSlider.max = data.total_frames - 1;
        startFrameSlider.value = 0;
        endFrameSlider.max = data.total_frames - 1;
        endFrameSlider.value = data.total_frames - 1;
        startFrameValue.textContent = '1';
        endFrameValue.textContent = data.total_frames;

        videoPreviewArea.classList.remove('hidden');
        notify('Video uploaded and frames extracted.', 'success');
    } catch (err) {
        notify(err.message, 'error');
        uploadZone.classList.remove('hidden');
    } finally {
        uploadLoading.classList.add('hidden');
    }
}

// ======== Step 2: Mask Annotation ========
function setupStep2() {
    // Frame sliders
    startFrameSlider.addEventListener('input', () => {
        startFrameValue.textContent = parseInt(startFrameSlider.value) + 1;
    });
    startFrameSlider.addEventListener('change', async () => {
        const frame = parseInt(startFrameSlider.value);
        startFrameValue.textContent = frame + 1;
        await selectFrame(frame);
        loadFrame(frame);
    });

    endFrameSlider.addEventListener('input', () => {
        endFrameValue.textContent = parseInt(endFrameSlider.value) + 1;
    });
    endFrameSlider.addEventListener('change', async () => {
        const frame = parseInt(endFrameSlider.value);
        endFrameValue.textContent = frame + 1;
        await fetch('/api/set-end-frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sid: state.sid, frame_number: frame }),
        });
    });

    // Click mode toggles
    $('#btn-positive').addEventListener('click', () => setClickMode('positive'));
    $('#btn-negative').addEventListener('click', () => setClickMode('negative'));

    // Canvas click
    maskCanvas.addEventListener('click', handleCanvasClick);

    // Mask actions
    $('#btn-add-mask').addEventListener('click', addMask);
    $('#btn-clear-clicks').addEventListener('click', clearClicks);
    $('#btn-remove-masks').addEventListener('click', removeMasks);
}

function setClickMode(mode) {
    state.clickMode = mode;
    $('#btn-positive').classList.toggle('active', mode === 'positive');
    $('#btn-negative').classList.toggle('active', mode === 'negative');
}

async function selectFrame(frameNumber) {
    await fetch('/api/select-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sid: state.sid, frame_number: frameNumber }),
    });
}

function loadFrame(frameNumber) {
    canvasLoading.classList.remove('hidden');
    const img = new Image();
    img.onload = () => {
        state.imageNaturalWidth = img.naturalWidth;
        state.imageNaturalHeight = img.naturalHeight;

        // Scale canvas to fit container while preserving aspect ratio
        const container = maskCanvas.parentElement;
        const maxW = container.clientWidth - 2;
        const maxH = 500;
        const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1);
        maskCanvas.width = Math.round(img.naturalWidth * scale);
        maskCanvas.height = Math.round(img.naturalHeight * scale);
        ctx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);
        canvasLoading.classList.add('hidden');
    };
    img.onerror = () => {
        canvasLoading.classList.add('hidden');
    };
    img.src = `/api/frame/${frameNumber}?sid=${state.sid}&t=${Date.now()}`;
}

async function handleCanvasClick(e) {
    const rect = maskCanvas.getBoundingClientRect();
    const scaleX = state.imageNaturalWidth / maskCanvas.width;
    const scaleY = state.imageNaturalHeight / maskCanvas.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    const label = state.clickMode === 'positive' ? 1 : 0;

    // Draw click indicator
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;
    ctx.beginPath();
    ctx.arc(canvasX, canvasY, 5, 0, Math.PI * 2);
    ctx.fillStyle = label === 1 ? '#22c55e' : '#ef4444';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    canvasLoading.classList.remove('hidden');

    try {
        const res = await fetch('/api/sam-click', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sid: state.sid, x, y, label }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'SAM click failed');

        // Reload the painted frame
        loadFrame(parseInt(startFrameSlider.value));
    } catch (err) {
        notify(err.message, 'error');
        canvasLoading.classList.add('hidden');
    }
}

async function addMask() {
    try {
        const res = await fetch('/api/add-mask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sid: state.sid }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to add mask');
        state.masks = data.masks;
        updateMaskList();
        loadFrame(parseInt(startFrameSlider.value));
        notify('Mask added.', 'success');
    } catch (err) {
        notify(err.message, 'error');
    }
}

async function clearClicks() {
    try {
        await fetch('/api/clear-clicks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sid: state.sid }),
        });
        loadFrame(parseInt(startFrameSlider.value));
    } catch (err) {
        notify(err.message, 'error');
    }
}

async function removeMasks() {
    try {
        await fetch('/api/remove-masks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sid: state.sid }),
        });
        state.masks = [];
        updateMaskList();
        loadFrame(parseInt(startFrameSlider.value));
        notify('All masks removed.', 'info');
    } catch (err) {
        notify(err.message, 'error');
    }
}

const maskColors = ['#4f6ef7', '#22c55e', '#f59e0b', '#ef4444', '#a855f7', '#ec4899'];

function updateMaskList() {
    if (state.masks.length === 0) {
        maskList.innerHTML = '<p class="mask-list-empty">No masks added yet</p>';
        return;
    }
    maskList.innerHTML = state.masks
        .map(
            (name, i) =>
                `<span class="mask-chip"><span class="dot" style="background: ${maskColors[i % maskColors.length]}"></span>${name}</span>`
        )
        .join('');
}

// ======== Step 3: Track & Inpaint ========
function setupStep3() {
    $('#btn-track').addEventListener('click', runTracking);
    $('#btn-inpaint').addEventListener('click', runInpainting);
}

function populateStep3Masks() {
    const container = $('#step3-mask-checkboxes');
    if (state.masks.length === 0) {
        container.innerHTML = '<span style="color: var(--text-secondary); font-size: 0.85rem;">No masks available</span>';
        return;
    }
    container.innerHTML = state.masks
        .map(
            (name, i) =>
                `<label><input type="checkbox" value="${name}" checked> <span class="dot" style="background: ${maskColors[i % maskColors.length]}; display: inline-block; width: 8px; height: 8px; border-radius: 50%;"></span> ${name}</label>`
        )
        .join('');
}

function getSelectedMasks() {
    const checkboxes = $$('#step3-mask-checkboxes input[type="checkbox"]:checked');
    return Array.from(checkboxes).map((cb) => cb.value);
}

async function runTracking() {
    const btn = $('#btn-track');
    const progress = $('#tracking-progress');
    btn.disabled = true;
    progress.classList.remove('hidden');

    try {
        const masks = getSelectedMasks();
        const res = await fetch('/api/track', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sid: state.sid, mask_selection: masks }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Tracking failed');

        // Show result
        $('#results-area').classList.remove('hidden');
        const trackingVideo = $('#tracking-video');
        trackingVideo.src = data.video + '&t=' + Date.now();
        trackingVideo.load();

        state.trackingDone = true;
        $('#btn-inpaint').disabled = false;
        notify('Tracking completed. You can now run inpainting.', 'success');
    } catch (err) {
        notify(err.message, 'error');
    } finally {
        btn.disabled = false;
        progress.classList.add('hidden');
    }
}

async function runInpainting() {
    if (!state.trackingDone) {
        notify('Run tracking first before inpainting.', 'error');
        return;
    }

    const btn = $('#btn-inpaint');
    const progress = $('#inpaint-progress');
    btn.disabled = true;
    progress.classList.remove('hidden');

    try {
        const masks = getSelectedMasks();
        const res = await fetch('/api/inpaint', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sid: state.sid,
                mask_selection: masks,
                resize_ratio: parseFloat(settingSliders.resize.input.value),
                dilate_radius: parseInt(settingSliders.dilate.input.value),
                raft_iter: parseInt(settingSliders.raftIter.input.value),
                subvideo_length: parseInt(settingSliders.subvideo.input.value),
                neighbor_length: parseInt(settingSliders.neighbor.input.value),
                ref_stride: parseInt(settingSliders.refStride.input.value),
            }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Inpainting failed');

        const inpaintVideo = $('#inpainting-video');
        inpaintVideo.src = data.video + '&t=' + Date.now();
        inpaintVideo.load();

        const downloadBtn = $('#btn-download');
        downloadBtn.href = data.video;
        downloadBtn.download = 'inpainted_video.mp4';
        downloadBtn.style.display = 'flex';

        notify('Inpainting completed.', 'success');
    } catch (err) {
        notify(err.message, 'error');
    } finally {
        btn.disabled = false;
        progress.classList.add('hidden');
    }
}

// ======== Settings sliders ========
function setupSettings() {
    Object.values(settingSliders).forEach(({ input, display }) => {
        input.addEventListener('input', () => {
            display.textContent =
                input.step && parseFloat(input.step) < 1
                    ? parseFloat(input.value).toFixed(2)
                    : input.value;
        });
    });
}
