
export class Viewport {
    constructor(targetWidth, targetHeight) {
        this.targetWidth = targetWidth;
        this.targetHeight = targetHeight;
        this.updateSize();
        this.mouseX = 0;
        this.mouseY = 0;
    }

    updateMouse() {
        this.mouseX = (mouseX - width / 2) / this.scaleFactor + this.targetWidth / 2;
        this.mouseY = (mouseY - height / 2) / this.scaleFactor + this.targetHeight / 2;
    }

    updateSize() {
        this.scaleFactor = min(width / this.targetWidth, height / this.targetHeight);
        this.fullW = width / this.scaleFactor;
        this.fullH = height / this.scaleFactor;
        this.fullX = this.targetWidth / 2 - this.fullW / 2;
        this.fullY = this.targetHeight / 2 - this.fullH / 2;
    }

    set() {
        push();
        translate(width / 2, height / 2);
        scale(this.scaleFactor, this.scaleFactor);
        translate(-this.targetWidth / 2, -this.targetHeight / 2);
    }

    reset() {
        pop();
    }
}
