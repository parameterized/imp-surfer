
import * as tf from '@tensorflow/tfjs';
import { SineActivation } from './customLayers.js';
import { transforms, targetTensors } from './sceneViewer.js';
import { utils } from './utils.js';

export let dispose = x => tf.dispose(x);

export class SceneModel {
    constructor(viewer) {
        this.viewer = viewer;

        this.model = tf.sequential();
        let filtersPerLayer = [32, 64, 32];
        let uniform = tf.initializers.randomUniform;
        let biasInit = uniform({ minval: -PI, maxval: PI });
        let limit = 2; // sqrt(6 / 6) * 2
        this.model.add(tf.layers.conv2d({
            filters: filtersPerLayer[0], kernelSize: 1, biasInitializer: biasInit, inputShape: [null, null, 6],
            kernelInitializer: uniform({ minval: -limit, maxval: limit })
        }));
        this.model.add(new SineActivation());
        for (let i = 1; i < filtersPerLayer.length; i++) {
            this.model.add(tf.layers.conv2d({
                filters: filtersPerLayer[i], kernelSize: 1, biasInitializer: biasInit,
                kernelInitializer: 'heUniform'
            }));
            this.model.add(new SineActivation());
        }
        this.model.add(tf.layers.conv2d({
            filters: 4, kernelSize: 1, biasInitializer: biasInit,
            kernelInitializer: 'heUniform', activation: 'tanh'
        }));

        // build model
        let x = tf.zeros([1, 1, 1, 6]);
        tf.dispose([this.model.predict(x), x]);

        this.optimizer = tf.train.adam(0.01);

        // test model
        this.image = createImage(64, 64);
        this.image.loadPixels();
        this.updateView();
    }

    async setImage(tensor) {
        let rescaledTensor = tf.tidy(() => tf.clipByValue(tensor.add(1).div(2), 0, 1));
        let newImage = await tf.browser.toPixels(rescaledTensor);
        tf.dispose([tensor, rescaledTensor]);
        utils.copyPixels(newImage, this.image.pixels);
        this.image.updatePixels();
    }

    getInput(mat4, jitter) {
        if (!mat4) {
            let c = this.viewer.cam;
            let a = c._getLocalAxes();
            mat4 = [...a.x, 0, ...a.y, 0, ...a.z, 0, c.eyeX, c.eyeY, c.eyeZ, 1];
        }
        let res = 64;
        return tf.tidy(() => {
            let camMatrix = tf.tensor(mat4, [4, 4]);
            let rayOrigins = camMatrix.slice([3, 0], [1, 3]).reshape([1, 1, -1]).tile([res, res, 1]);

            // todo: non-square aspect ratio
            let maxCoord = 1 - 1 / res;
            maxCoord *= tan(this.viewer.angleX);
            let rx = tf.linspace(-maxCoord, maxCoord, res);
            let ry = tf.linspace(-maxCoord, maxCoord, res);
            rx = rx.expandDims(0).tile([res, 1]);
            ry = ry.expandDims(1).tile([1, res]);
            let rxy = tf.stack([rx, ry], -1);
            if (jitter) { // subpixel randomization
                rxy = rxy.add(tf.randomUniform(rxy.shape, -1 / res, 1 / res));
            }
            let rz = tf.onesLike(rx).mul(-1).expandDims(-1);

            // transform and normalize ray directions
            let rayDirs = tf.concat([rxy, rz], -1);
            rayDirs = rayDirs.reshape([-1, 3]);
            rayDirs = rayDirs.matMul(camMatrix.slice(0, [3, 3]));
            rayDirs = rayDirs.reshape([res, res, 3]);
            rayDirs = rayDirs.div(rayDirs.norm('euclidean', -1, true));

            // localize rays by using closest point to object as origin
            // (object at 0,0,0)
            let t = rayOrigins.mul(-1).mul(rayDirs).sum(-1, true);
            let localOrigins = rayOrigins.add(rayDirs.mul(t));

            return tf.concat([localOrigins, rayDirs], -1);
        });
    }

    predict(x) {
        return this.model.predict(x.expandDims(0)).squeeze(0);
    }

    updateView(sameInput) {
        if (!sameInput) {
            tf.dispose(this.camViewInput);
            this.camViewInput = this.getInput();
        }
        this.setImage(this.predict(this.camViewInput));
    }

    trainStep() {
        if (targetTensors.length === 0) { return; }
        let i = int(random(targetTensors.length));
        //i = 2;
        let y = targetTensors[i];
        this.optimizer.minimize(() => {
            let viewInput = this.getInput(transforms.frames[i].mat.mat4, true);
            let yPred = this.predict(viewInput);
            return tf.losses.meanSquaredError(y, yPred);
        });
    }
}
