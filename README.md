# ImageToDEM
Generating a 3D surface model from 2D satellite imagery


## Usage

## Visualization

  Visualization of a DEM file can be achieved in two ways.

  * By using this great [tool](https://github.com/zhunor/threejs-dem-visualizer).
    First run the DEM2rgb.py, then positiveDEM.py scripts.
    Then run `cd threejs-dem-visualizer`, `yarn install`, `yarn build` and `yarn dev`
    Project will be live @ `localhost:8080`

  * With pythons matplotlib functions plot_surface or scatter3D.
    First run the DEM2rgb.py script, then choose between running plot3D.py and scatter3D.py
