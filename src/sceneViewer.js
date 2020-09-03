
import * as tf from '@tensorflow/tfjs';
import { UI } from './ui.js';
import { SceneModel, dispose } from './sceneModel.js';
import { targetWidth, targetHeight, gfx } from './index.js';
import { utils } from './utils.js';

export let transforms;
let sortedTransforms = []; // for nearest view
export function loadTransforms() {
    transforms = loadJSON('data/lego/transforms_train.json', () => {
        for (let i = 0; i < transforms.frames.length; i++) {
            let mat2d = transforms.frames[i].transform_matrix;
            // convert to p5 matrix
            let mlist = [];
            for (let j = 0; j < 16; j++) {
                mlist.push(mat2d[j % 4][floor(j / 4)]);
            }
            let mat = new p5.Matrix();
            mat.set(new Float32Array(mlist));
            let fixAxes = new p5.Matrix();
            fixAxes.set(new Float32Array([
                1, 0, 0, 0,
                0, 0, 1, 0,
                0, -1, 0, 0,
                0, 0, 0, 1
            ]));
            mat.mult(fixAxes);
            mat.scale(1, -1, 1);
            transforms.frames[i].mat = mat;

            let t = { mat: mat, id: i };
            // precompute cam to origin angles
            let d = sqrt(sq(mat2d[0][3]) + sq(mat2d[1][3]) + sq(mat2d[2][3]));
            t.cto = { x: -mat2d[0][3] / d, y: mat2d[2][3] / d, z: -mat2d[1][3] / d };
            sortedTransforms.push(t);
        }
    });
}

export let targetTensors = [];

export class SceneViewer {
    aspectRatio = 1;
    angleX = transforms.camera_angle_x; // half fov
    
    dist1 = 0.5; // for showing all cameras
    dist2 = 4; // for nearest view

    viewMode = 0; // all cameras, nearest view, predicted view
    selectedView = 0;

    orbit = false;
    orbitT = 0;

    trainStepN = 0;
    trainStepMax = 100;

    constructor() {
        this.webglGraphics = createGraphics(800, 800, WEBGL);
        let g = this.webglGraphics;
        this.cam = g.createCamera();
        this.yaw = 0;
        this.pitch = QUARTER_PI;
        let c = this.cam;
        c.perspective(2 * this.angleX, 1, 0.1, 10000);
        c.setPosition(0, -5, 5);
        c.lookAt(0, 0, 0);

        this.model = new SceneModel(this);

        this.ui = new UI();
        this.load10Btn = this.ui.addButton({
            text: 'Load 10 Images', box: [50, 60, 300, 60],
            action: () => this.loadImages(10)
        });
        this.loadAllBtn = this.ui.addButton({
            text: 'Load All Images', box: [50, 140, 300, 60],
            action: () => this.loadImages()
        });
    }

    loadImages(n) {
        if (this.loadingImages) { return; }
        this.loadingImages = true;
        let nImgs = 100;
        n = n === undefined ? nImgs : n;
        n = min(n, nImgs - gfx.data.length);
        // load asynchronously but in order
        let f = x => {
            if (x > 0) {
                loadImage(`data/lego/train/r_${gfx.data.length}.png`, img => {
                    gfx.data.push(img);

                    let res = 64;
                    let scaledImage = img.get();
                    scaledImage.resize(res, res);
                    scaledImage.loadPixels();
                    targetTensors.push(tf.tidy(() => {
                        return tf.browser.fromPixels({
                            data: Uint8Array.from(scaledImage.pixels),
                            width: res, height: res
                        }, 4).mul(2 / 255).sub(1); // [-1, 1]
                    }));

                    f(x - 1);
                });
            } else {
                this.loadingImages = false;
                // remove button when no longer possible
                let remainingImgs = nImgs - gfx.data.length;
                if (remainingImgs < 10) {
                    let i = this.ui.buttons.indexOf(this.load10Btn);
                    if (i > -1) {
                        this.ui.buttons.splice(i, 1);
                    }
                }
                if (remainingImgs <= 0) {
                    let i = this.ui.buttons.indexOf(this.loadAllBtn);
                    if (i > -1) {
                        this.ui.buttons.splice(i, 1);
                    }
                }
            }
        };
        f(n);
    }

    mousePressed() {
        this.ui.mousePressed();
        if (mouseButton === RIGHT && utils.mouseInRect(targetWidth / 2 - 400, targetHeight / 2 - 400, 800, 800)) {
            utils.requestPointerLock();
            this.movingCamera = true;
        }
    }

    mouseReleased() {
        if (mouseButton === RIGHT) {
            utils.exitPointerLock();
            this.movingCamera = false;
        }
    }

    mouseMoved(event) {
        if (this.movingCamera) {
            this.yaw -= event.movementX / 200;
            this.pitch = constrain(this.pitch + event.movementY / 200, -HALF_PI + 0.001, HALF_PI - 0.001);

            let zp = cos(this.pitch);
            let x = -zp * sin(this.yaw);
            let y = sin(this.pitch);
            let z = -zp * cos(this.yaw);
            
            let c = this.cam;
            c.lookAt(c.eyeX + x, c.eyeY + y, c.eyeZ + z);

            if (this.viewMode === 2) {
                this.doUpdateView = true;
                this.viewWasMoved = true;
            }
        }
    }

    updateYawAndPitch() {
        let z = this.cam._getLocalAxes().z;
        this.yaw = -(atan2(-z[2], -z[0]) + HALF_PI);
        this.pitch = asin(-z[1]);
    }

    keyPressed() {
        switch (keyCode) {
            case 9: // Tab
                this.viewMode = (this.viewMode + 1) % 3;
                if (this.viewMode === 1) {
                    this.sortViews();
                } else if (this.viewMode === 2) {
                    this.model.updateView();
                }
                break;
            case 49: // 1
                if (gfx.data.length > 0) {
                    this.selectedView %= gfx.data.length;
                    let m = transforms.frames[this.selectedView].mat.mat4;
                    let x = m[12], y = m[13], z = m[14];
                    let cd = createVector(-m[8], -m[9], -m[10]);
                    this.cam.camera(x, y, z, x + cd.x, y + cd.y, z + cd.z, 0, 1, 0);
                    this.updateYawAndPitch();

                    if (this.viewMode === 1) {
                        this.sortViews();
                    } else if (this.viewMode === 2) {
                        this.doUpdateView = true;
                        this.viewWasMoved = true;
                    }

                    this.selectedView++;
                }
                break;
            case 79: // O
                this.orbit = !this.orbit;
                break;
            case 82: // R
                dispose(this.model);
                this.model = new SceneModel(this);
                this.model.updateView();
                break;
            case 84: // T
                this.trainStepN = 0;
                if (keyIsDown(16)) { // Shift
                    this.trainStepMax = 1000;
                }
                break;
        }
    }

    sortViews() {
        let c = this.cam;
        let d = sqrt(sq(c.eyeX) + sq(c.eyeY) + sq(c.eyeZ));
        let cto = { x: -c.eyeX / d, y: -c.eyeY / d, z: -c.eyeZ / d }; // cam to origin
        // assume all cameras looking at origin
        sortedTransforms.sort((a, b) => {
            let ad = sq(a.cto.x - cto.x) + sq(a.cto.y - cto.y) + sq(a.cto.z - cto.z);
            let bd = sq(b.cto.x - cto.x) + sq(b.cto.y - cto.y) + sq(b.cto.z - cto.z);
            return ad - bd;
        });
    }

    update(dt) {
        let c = this.cam;
        if (this.orbit) {
            let t = (cos(this.orbitT * 0.6) + 1) / 2;
            let p = p5.Vector.fromAngles(lerp(HALF_PI * 0.2, HALF_PI * 0.9, t), this.orbitT, 4);
            c.setPosition(p.x, p.y, p.z);
            c.lookAt(0, 0, 0);
            this.updateYawAndPitch();
            this.viewWasMoved = true;
            this.orbitT += dt;
        } else if (this.movingCamera) {
            let dx = Number(keyIsDown(68)) - Number(keyIsDown(65)); // D/A
            let dy = -Number(keyIsDown(32)); // Space
            let dz = Number(keyIsDown(83)) - Number(keyIsDown(87)); // S/W
            let s = 4 * (1 + Number(keyIsDown(16)) * 2) * dt; // Shift
            c.move(dx * s, 0, dz * s);
            c.setPosition(c.eyeX, c.eyeY + dy * s, c.eyeZ); // dy in world space

            if (abs(dx) + abs(dy) + abs(dz) > 0) {
                this.viewWasMoved = true;
            }
        }
        if (this.viewWasMoved) {
            if (this.viewMode === 1) {
                this.sortViews();
            } else if (this.viewMode === 2) {
                this.doUpdateView = true;
            }
        }

        if (gfx.data.length === transforms.frames.length) {
            if (this.trainStepN < this.trainStepMax) {
                this.model.trainStep();
                this.trainStepN++;
                if (this.viewMode === 2) {
                    this.doUpdateView = true;
                }
            } else {
                this.trainStepMax = 100;
            }
        }

        if (this.doUpdateView) {
            this.model.updateView(!this.viewWasMoved);
        }
        this.doUpdateView = false;
        this.viewWasMoved = false;
    }

    draw() {
        let g = this.webglGraphics;
        g.clear(0);
        g.background(0);
        g.normalMaterial();
        switch (this.viewMode) {
            case 0: // all cameras
                // origin
                g.push();
                g.scale(1 / 10);
                g.sphere(1);
                g.cylinder(0.5, 4);
                g.rotateX(HALF_PI);
                g.cylinder(0.5, 4);
                g.rotateZ(HALF_PI);
                g.cylinder(0.5, 4);
                g.pop();

                // image viewpoints
                for (let i = 0; i < gfx.data.length; i++) {
                    g.push();
                    g.applyMatrix(...transforms.frames[i].mat.mat4);

                    // camera 
                    g.push();
                    g.scale(1 / 10);
                    g.sphere(1);
                    g.rotateX(HALF_PI);
                    g.translate(0, 0, 2);
                    g.cylinder(0.2, 1.6);
                    g.translate(0, -0.6, 0);
                    g.sphere(0.4);
                    g.pop();

                    // image
                    g.translate(0, 0, -this.dist1);
                    g.texture(gfx.data[i]);
                    let w = tan(this.angleX) * this.dist1 * 2;
                    g.plane(w, w / this.aspectRatio);

                    g.pop();
                }
                break;
            case 1: // nearest view
                if (gfx.data.length > 0) {
                    g.push();
                    let t = sortedTransforms.find(v => v.id < gfx.data.length);
                    g.applyMatrix(...t.mat.mat4);
                    g.translate(0, 0, -this.dist2);
                    g.texture(gfx.data[t.id]);
                    let w = tan(this.angleX) * this.dist2 * 2;
                    g.plane(w, w / this.aspectRatio);
                    g.pop();
                }
                break;
        }

        push();
        translate(targetWidth / 2, targetHeight / 2);
        imageMode(CENTER);
        image(g, 0, 0);
        if (this.viewMode === 2) {
            image(this.model.image, 0, 0, 800, 800);
        }
        if (this.trainStepN < this.trainStepMax) {
            fill('#AACADC');
            rect(-400, 420, (this.trainStepN / this.trainStepMax) * 800, 30);
        }
        pop();

        this.ui.draw();
    }
}
