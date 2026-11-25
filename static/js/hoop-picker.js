/* BallCut - Hoop region picker, shared by the home page and the review page
   Shows a frame on a canvas and lets the user drag a box over it with the mouse or a finger.
   The box is reported in video pixels, so it does not depend on how large the canvas is displayed. */

function createHoopPicker(canvas, options) {
    const wrapper = options.wrapper || canvas.parentElement;
    const hint = options.hint || null;
    const onChange = options.onChange || function () {};
    const ctx = canvas.getContext('2d');

    let videoWidth = 0, videoHeight = 0;
    let frame = null;          // the frame without the box
    let region = null;         // [x1, y1, x2, y2] in video pixels
    let start = null, current = null;   // box being dragged, in canvas pixels

    function drawBox(x, y, w, h) {
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(0, 255, 136, 0.15)';
        ctx.fillRect(x, y, w, h);
    }

    function draw() {
        if (!frame) return;
        ctx.drawImage(frame, 0, 0, canvas.width, canvas.height);
        if (start && current) {
            drawBox(Math.min(start.x, current.x), Math.min(start.y, current.y),
                    Math.abs(current.x - start.x), Math.abs(current.y - start.y));
        } else if (region) {
            const sx = canvas.width / videoWidth, sy = canvas.height / videoHeight;
            drawBox(region[0] * sx, region[1] * sy, (region[2] - region[0]) * sx, (region[3] - region[1]) * sy);
        }
    }

    // Canvas pixels -> video pixels, kept inside the frame
    function toVideo(p) {
        return {
            x: Math.max(0, Math.min(canvas.width, p.x)) * videoWidth / canvas.width,
            y: Math.max(0, Math.min(canvas.height, p.y)) * videoHeight / canvas.height
        };
    }

    function pointOf(e) {
        const rect = canvas.getBoundingClientRect();
        const p = (e.touches && e.touches[0]) || e;
        // The canvas can be displayed smaller than its pixel size
        return {
            x: (p.clientX - rect.left) * canvas.width / rect.width,
            y: (p.clientY - rect.top) * canvas.height / rect.height
        };
    }

    function down(e) {
        e.preventDefault();
        start = current = pointOf(e);
        region = null;
        if (hint) hint.classList.add('hidden');
    }

    function move(e) {
        if (!start) return;
        e.preventDefault();
        current = pointOf(e);
        draw();
    }

    function up() {
        if (!start) return;
        const a = toVideo(start), b = toVideo(current);
        start = current = null;
        const box = [Math.round(Math.min(a.x, b.x)), Math.round(Math.min(a.y, b.y)),
                     Math.round(Math.max(a.x, b.x)), Math.round(Math.max(a.y, b.y))];
        region = (box[2] - box[0] < 20 || box[3] - box[1] < 20) ? null : box;
        draw();
        onChange(region);
    }

    canvas.addEventListener('mousedown', down);
    canvas.addEventListener('mousemove', move);
    canvas.addEventListener('mouseup', up);
    canvas.addEventListener('mouseleave', up);
    canvas.addEventListener('touchstart', down, {passive: false});
    canvas.addEventListener('touchmove', move, {passive: false});
    canvas.addEventListener('touchend', up);

    return {
        // Shows a frame (imageUrl) of a video of the given size; any box shown before is cleared.
        // onReady() is called once the frame is displayed, onError() if it could not be loaded.
        load(imageUrl, width, height, onReady, onError) {
            videoWidth = width;
            videoHeight = height;
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = function () {
                // Scale to fit the wrapper (max 960 px wide)
                const available = wrapper.clientWidth > 20 ? wrapper.clientWidth - 20 : 960;
                const scale = Math.min(960, available) / img.width;
                canvas.width = Math.round(img.width * scale);
                canvas.height = Math.round(img.height * scale);
                frame = img;
                region = start = current = null;
                draw();
                if (onReady) onReady();
            };
            img.onerror = function () {
                frame = null;
                if (onError) onError();
            };
            img.src = imageUrl;
        },
        // Shows a box that was chosen earlier
        setRegion(box) {
            region = box || null;
            start = current = null;
            draw();
        },
        // The box shown right now, in video pixels, or null
        getRegion() {
            return region;
        }
    };
}
