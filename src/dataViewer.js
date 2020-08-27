
import { UI } from './ui.js';
import { targetWidth, targetHeight, gfx } from './index.js';
import { utils } from './utils.js';

let transforms;
let sortedTransforms = []; // for nearest view
export function loadTransforms() {
    transforms = loadJSON('data/lego/transforms_train.json', () => {
        for (let i = 0; i < transforms.frames.length; i++) {
            let t = { mat: transforms.frames[i].transform_matrix, id: i };
            // precompute cam to origin angles
            let d = sqrt(sq(t.mat[0][3]) + sq(t.mat[1][3]) + sq(t.mat[2][3]));
            t.cto = { x: -t.mat[0][3] / d, y: t.mat[2][3] / d, z: -t.mat[1][3] / d };
            sortedTransforms.push(t);
        }
    });
}

export class DataViewer {
    aspectRatio = 1;
    angleX = transforms.camera_angle_x; // half fov
    
    dist1 = 0.5; // for showing all cameras
    dist2 = 4; // for nearest view
    predict = false;

    constructor() {
        this.g = createGraphics(800, 800, WEBGL);
        let g = this.g;
        this.cam = g.createCamera();
        this.yaw = 0;
        this.pitch = QUARTER_PI;
        let c = this.cam;
        c.perspective(HALF_PI, 1, 0.1, 10000);
        c.setPosition(0, -5, 5);
        c.lookAt(0, 0, 0);

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
        if (utils.mouseInRect(targetWidth / 2 - 400, targetHeight / 2 - 400, 800, 800) && mouseButton === RIGHT) {
            utils.requestPointerLock();
            this.movingCamera = true;
        }
    }

    mouseReleased() {
        utils.exitPointerLock();
        this.movingCamera = false;
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
        }
    }

    keyPressed() {
        if (keyCode === 9) { // Tab
            this.predict = !this.predict;
        }
    }

    update(dt) {
        if (this.movingCamera) {
            let dx = Number(keyIsDown(68)) - Number(keyIsDown(65)); // D/A
            let dy = -Number(keyIsDown(32)); // Space
            let dz = Number(keyIsDown(83)) - Number(keyIsDown(87)); // S/W
            let s = 4 * (1 + Number(keyIsDown(16)) * 2) * dt; // Shift
            let c = this.cam;
            c.move(dx * s, 0, dz * s);
            c.setPosition(c.eyeX, c.eyeY + dy * s, c.eyeZ); // dy in world space

            // resort by closest to camera if moved
            if (abs(dx) + abs(dy) + abs(dz) > 0) {
                let d = sqrt(sq(c.eyeX) + sq(c.eyeY) + sq(c.eyeZ));
                let cto = { x: -c.eyeX / d, y: -c.eyeY / d, z: -c.eyeZ / d }; // cam to origin
                // assume all cameras looking at origin
                sortedTransforms.sort((a, b) => {
                    let ad = sq(a.cto.x - cto.x) + sq(a.cto.y - cto.y) + sq(a.cto.z - cto.z);
                    let bd = sq(b.cto.x - cto.x) + sq(b.cto.y - cto.y) + sq(b.cto.z - cto.z);
                    return ad - bd;
                });
            }
        }
    }

    draw() {
        let g = this.g;
        g.clear(0);
        g.background(0);
        g.normalMaterial();

        // data is z-up
        g.push();
        g.rotateX(HALF_PI);

        if (this.predict) {
            if (gfx.data.length > 0) {    
                g.push();
                let t = sortedTransforms.find(v => v.id < gfx.data.length);
                let mlist = [];
                for (let j = 0; j < 16; j++) {
                    mlist.push(t.mat[j % 4][floor(j / 4)]);
                }
                g.applyMatrix(...mlist);
                g.scale(1, -1, 1);
                g.translate(0, 0, -this.dist2);
                g.texture(gfx.data[t.id]);
                let w = tan(this.angleX) * this.dist2 * 2;
                g.plane(w, w / this.aspectRatio);
                g.pop();
            }
        } else {
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
                let m = transforms.frames[i].transform_matrix;
                let mlist = [];
                for (let j = 0; j < 16; j++) {
                    mlist.push(m[j % 4][floor(j / 4)]);
                }
                g.applyMatrix(...mlist);
                g.scale(1, -1, 1);
                
                // camera 
                g.push();
                g.scale(1 / 10);
                g.sphere(1);
                g.translate(0, 0, -2);
                g.cylinder(0.2, 2);
                g.push();
                g.translate(0, -1, 0);
                g.sphere(0.4);
                g.pop();
                g.pop();
                
                // image
                g.translate(0, 0, -this.dist1);
                g.texture(gfx.data[i]);
                let w = tan(this.angleX) * this.dist1 * 2;
                g.plane(w, w / this.aspectRatio);
    
                g.pop();
            }
        }

        g.pop();

        push();
        translate(targetWidth / 2, targetHeight / 2);
        imageMode(CENTER);
        image(g, 0, 0);
        pop();

        this.ui.draw();
    }
}
