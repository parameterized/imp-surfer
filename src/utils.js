
import { viewport, touchTimer } from './index.js';

document.body.requestPointerLock = (document.body.requestPointerLock || document.body.mozRequestPointerLock || document.body.webkitRequestPointerLock);
document.exitPointerLock = (document.exitPointerLock || document.mozExitPointerLock || document.webkitExitPointerLock);

let utils = {
    mouseInRect(r_or_x, y, w, h) {
        let x = r_or_x;
        if (typeof (r_or_x) === 'object') {
            let r = r_or_x;
            if (r.length) {
                x = r[0]; y = r[1]; w = r[2]; h = r[3];
            } else {
                x = r.x; y = r.y; w = r.w; h = r.h;
            }
        }
        return (viewport.mouseX > x && viewport.mouseX < x + w
            && viewport.mouseY > y && viewport.mouseY < y + h);
    },
    cursorStyle(style) {
        if (touchTimer > 0.5) {
            document.body.style.cursor = style;
        }
    },
    setPointer() {
        utils.cursorStyle('pointer');
    },
    requestPointerLock() {
        document.body.requestPointerLock();
    },
    exitPointerLock() {
        document.exitPointerLock();
    },
    hash(x, y) {
        return abs(sin(x * 12.9898 + y * 4.1414) * 43758.5453) % 1;
    },
    pingPong(t) {
        t = abs(t);
        return t % 2 < 1 ? t % 1 : 1 - t % 1;
    },
    mod(n, m) {
        return ((n % m) + m) % m;
    },
    deltaAngle(a, b) {
        let d = b - a;
        return utils.mod(d + PI, TWO_PI) - PI;
    },
    lerpAngle(a, b, t) {
        let theta = b - a;
        if (theta > PI) {
            a += TWO_PI;
        } else if (theta < -PI) {
            a -= TWO_PI;
        }
        return lerp(a, b, t);
    },
    copyPixels(src) {
        // https://forum.processing.org/two/discussion/10485/troubles-with-the-p5-image-pixels-array
        const args = arguments.length;
        var sIdx = 0, dst = arguments[1], dIdx = 0,
            len = args == 3 ? ~~Math.abs(arguments[2]) : src.length;
        if (args > 3) {
            sIdx = ~~Math.abs(dst);
            dst = arguments[2];
            dIdx = ~~Math.abs(arguments[3]);
            len = args > 4 ? ~~Math.abs(arguments[4]) : len;
        }
        const sLen = src.length, dLen = dst.length, end = Math.min(len + sIdx, sLen);
        if (!sIdx && sLen <= len && sLen + dIdx <= dLen && ArrayBuffer.isView(dst))
            dst.set(src, dIdx);
        else
            for (var i = sIdx, j = dIdx; i < end & j < dLen; dst[j++] = src[i++]);
        return dst;
    },
    getBoxFromCenter(x, y, w, h) {
        return [x - w / 2, y - h / 2, w, h];
    },
    sum(a) {
        return a.reduce((acc, cv) => acc + cv, 0);
    },
    prod(a) {
        return a.reduce((acc, cv) => acc * cv);
    }
};

let ease = {
    inQuad(t) { return t * t },
    outQuad(t) { return t * (2 - t) },
    inOutQuad(t) { return t < .5 ? 2 * t * t : -1 + (4 - 2 * t) * t },
    inCubic(t) { return t * t * t },
    outCubic(t) { return (--t) * t * t + 1 },
    inOutCubic(t) { return t < .5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1 },
    inQuart(t) { return t * t * t * t },
    outQuart(t) { return 1 - (--t) * t * t * t },
    inOutQuart(t) { return t < .5 ? 8 * t * t * t * t : 1 - 8 * (--t) * t * t * t },
    inQuint(t) { return t * t * t * t * t },
    outQuint(t) { return 1 + (--t) * t * t * t * t },
    inOutQuint(t) { return t < .5 ? 16 * t * t * t * t * t : 1 + 16 * (--t) * t * t * t * t }
};

export { utils, ease };
