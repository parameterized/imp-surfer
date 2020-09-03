
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
        let limit = 1.5; // sqrt(6 / 6) * 1.5
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

    getInput(mat4) {
        if (!mat4) {
            let c = this.viewer.cam;
            let a = c._getLocalAxes();
            mat4 = [...a.x, 0, ...a.y, 0, ...a.z, 0, c.eyeX, c.eyeY, c.eyeZ, 1];
            mat4[0] *= -1;
            mat4[1] *= -1;
            mat4[2] *= -1;
        }
        let res = 64;
        return tf.tidy(() => {
            let camMatrix = tf.tensor(mat4, [4, 4]);
            let rayOrigins = camMatrix.slice([3, 0], [1, 3]).reshape([1, 1, -1]).tile([res, res, 1]);

            let coords = [];
            let maxCoord = 1 - 1 / res;
            // todo: non-square aspect ratio
            maxCoord /= tan(this.viewer.angleX / 2);
            for (let axis = 0; axis < 2; axis++) {
                let pts = tf.linspace(-maxCoord, maxCoord, res);
                if (axis === 0) {
                    coords.push(pts.mul(-1).expandDims(0).tile([res, 1]));
                } else {
                    coords.push(pts.expandDims(1).tile([1, res]));
                }
            }
            let xy = tf.stack(coords, -1);
            // subpixel randomization
            xy = xy.add(tf.randomUniform(xy.shape, -1 / res, 1 / res));
            let z = tf.ones([res, res]).mul(-1).expandDims(-1);

            let rayDirs = tf.concat([xy, z], -1);
            rayDirs = rayDirs.reshape([-1, 3]);
            rayDirs = rayDirs.matMul(camMatrix.slice([0, 0], [3, 3]));
            rayDirs = rayDirs.reshape([res, res, 3]);
            if (true) {
                rayDirs = rayDirs.div(rayDirs.norm('euclidean', -1, true));
    
                // localize rays by using closest point to object as origin
                // (object at 0,0,0)
                let t = rayOrigins.mul(-1).mul(rayDirs).sum(-1, true);
                rayOrigins = rayOrigins.add(rayDirs.mul(t));
            } else {
                // project origins to plane intersecting object with camera normal (rayDirs not normalized)
                // (doesn't work as well)
                let cd = tf.tensor([0, 0, -1]).expandDims();
                cd = cd.matMul(camMatrix.slice([0, 0], [3, 3])).squeeze();
                let t = rayOrigins.slice(0, [1, 1]).squeeze().mul(-1).mul(cd).sum(-1);
                rayOrigins = rayOrigins.add(rayDirs.mul(t));
            }

            // test ability to learn a gradient by only inputting dist of projected origin to object
            //let d = rayOrigins.norm('euclidean', -1, true);
            //return tf.concat([d, tf.zeros([res, res, 5])], -1);

            return tf.concat([rayOrigins, rayDirs], -1);
        });
    }

    predict(x) {
        return tf.tidy(() => this.model.predict(x.expandDims(0)).squeeze());
    }

    updateView(sameInput) {
        if (!sameInput) {
            tf.dispose(this.viewInput);
            this.viewInput = this.getInput();
        }
        this.setImage(this.predict(this.viewInput));
    }

    trainStep() {
        if (targetTensors.length === 0) { return; }
        this.optimizer.minimize(() => {
            let i = int(random(targetTensors.length));
            //i = 0;
            let x = this.getInput(transforms.frames[i].mat.mat4);
            let y = targetTensors[i];
            let yPred = this.predict(x);
            return tf.losses.meanSquaredError(y, yPred);
        });
        this.setImage(this.predict(this.viewInput));
    }
}
