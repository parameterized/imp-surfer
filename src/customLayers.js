
import * as tf from '@tensorflow/tfjs';

export class SineActivation extends tf.layers.Layer {
    constructor(config) {
        super(config || {});
    }

    call(input) {
        return tf.sin(input[0]);
    }

    static get className() {
        return 'SineActivation';
    }
}
