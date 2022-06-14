# ImpSurfer

Implicit surface renderer for neural view synthesis

Trains a single sample ray -> pixel model by first shifting ray origins along the ray to minimize their distance from the center of the scene

## Running

The lastest version will be hosted on Github [here](https://parameterized.github.io/imp-surfer).

If you want to modify the code, install the dependencies (`npm install`) then `npm run dev-server` to start the webpack development server.

## Controls
- Right click and hold + Mouse / WASD / Space to move camera
- Hold Shift to move faster
- Tab to cycle through view modes (all views in dataset, closest view, predicted view)
- 1 to set your camera to a dataset view
- O to toggle automatic camera orbiting
- R to reset model weights
- T to train for 100 more steps, Shift + T to train for 1000

All views must be loaded to start training. The model will train for 100 steps when all views are loaded.

## Attribution

The lego data is from [NeRF](https://github.com/bmild/nerf), using [Heinzelnisse's model](https://www.blendswap.com/blend/11490)
