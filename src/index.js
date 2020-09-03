
import { Viewport } from './viewport.js';
import { SceneViewer, loadTransforms } from './sceneViewer.js';

export let targetWidth = 1600;
export let targetHeight = 900;
export let viewport;
export let gfx, sfx;

export let defaultVolume = 0.5;
export let touchUsed = false;
export let touchIsPressed = false;
export let touchTimer = 1; // disable mouse if touch use in last 0.5s
export let ptouches = []; // touches before callback
export let touch = null; // set to what callback is referencing

let fixedDt = 1 / 60;
let dtTimer = 0;

let sceneViewer;

window.preload = function () {
    gfx = {
        data: []
    };

    sfx = {};
    masterVolume(defaultVolume);

    loadTransforms();
}

window.setup = function () {
    let canvas = createCanvas(window.innerWidth, window.innerHeight);
    canvas.parent('sketch');

    // prevent default for right click, double click, and tab
    canvas.elt.addEventListener('contextmenu', e => {
        e.preventDefault();
    });
    canvas.elt.addEventListener('mousedown', e => {
        if (e.detail > 1) {
            e.preventDefault();
        }
    });
    document.addEventListener('keydown', e => {
        if (e.keyCode === 9) { // Tab
            e.preventDefault();
        }
    });

    strokeJoin(ROUND);
    smooth();

    viewport = new Viewport(targetWidth, targetHeight);
    sceneViewer = new SceneViewer();
    window.sceneViewer = sceneViewer;

    document.addEventListener('mousemove', e => sceneViewer.mouseMoved(e));
}

function pressed() {
    sceneViewer.mousePressed();
}
function released() {
    sceneViewer.mouseReleased();
}

window.mousePressed = function (event) {
    event.preventDefault();
    if (touchTimer > 0.5) {
        pressed();
    }
}
window.touchStarted = function (event) {
    event.preventDefault();
    touchUsed = true;
    // first element in touches that isn't in ptouches
    touch = touches.filter(t => ptouches.findIndex(pt => pt.id === t.id) === -1)[0];
    touchIsPressed = true;
    if (touch) {
        mouseX = touch.x;
        mouseY = touch.y;
        viewport.updateMouse();
    }
    pressed();
    ptouches = [...touches];
    touch = null;
}

window.mouseReleased = function (event) {
    event.preventDefault();
    // mouseButton not always accurate when multiple were down
    switch (event.button) {
        case 0:
            window.mouseButton = LEFT;
            break;
        case 1:
            window.mouseButton = CENTER;
            break;
        case 2:
            window.mouseButton = RIGHT;
            break;
    }
    if (touchTimer > 0.5) {
        released();
    }
}
window.touchEnded = function (event) {
    event.preventDefault();
    // first element in ptouches that isn't in touches
    touch = ptouches.filter(pt => touches.findIndex(t => t.id === pt.id) === -1)[0];
    if (touches.length === 0) {
        touchIsPressed = false;
    }
    released();
    ptouches = [...touches];
    touch = null;
}

window.mouseDragged = function (event) {
    event.preventDefault();
}
window.touchMoved = function (event) {
    event.preventDefault();
}
window.mouseWheel = function (event) {
    event.preventDefault();
}

window.keyPressed = function (event) {
    sceneViewer.keyPressed();
}

function update() {
    document.body.style.cursor = 'default';
    let dt = min(1 / frameRate(), 1 / 10);
    dtTimer += dt;
    while (dtTimer > 0) {
        dtTimer -= fixedDt;
        fixedUpdate(fixedDt);
    }

    viewport.updateMouse();
    if (touchIsPressed) {
        touchTimer = 0;
    } else {
        touchTimer += dt;
    }
}

function fixedUpdate(dt) {
    sceneViewer.update(dt);
}

window.draw = function () {
    update();
    noStroke();
    background('#1F1F1F');

    viewport.set();

    sceneViewer.draw();

    // cover top/bottom off-screen graphics
    fill('#1F1F1F');
    let v = viewport;
    rect(v.fullX, v.fullY, v.fullW, 0 - v.fullY);
    rect(v.fullX, v.targetHeight, v.fullW, v.fullY + v.fullH - v.targetHeight);
    // cover sides
    rect(v.fullX, v.fullY, 0 - v.fullX, v.fullH);
    rect(v.targetWidth, v.fullY, v.fullX + v.fullW - v.targetWidth, v.fullH);

    viewport.reset();
}

window.windowResized = function () {
    resizeCanvas(windowWidth, windowHeight);
    if (viewport) {
        viewport.updateSize();
    }
}
