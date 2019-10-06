# ImageToDEM
Generating a 3D surface model from 2D satellite imagery


## Usage

## Visualization

  Visualization of a DEM file can be achieved in two ways.

  * By using this great [tool](https://github.com/zhunor/threejs-dem-visualizer).
    Run `python3 Visualizer.py dem.tif image.jpg scale`

    Where:
     * `dem.tif` is a path to a DEM file in .tif format.

     * `image.jpg` is optional, a path to a .jpg image corresponding to the DEM
     image. If not given, the earth engine API is used to fetch one.

     * `scale` is optional, an integer to specify the resolution of the image
     fetched by the API. Bigger scale -> reduced resolution (default is 50).

    Project will be live @ `localhost:8080`

  * With pythons matplotlib functions plot_surface or scatter3D.
    First run the DEM2rgb.py script, then choose between running plot3D.py and scatter3D.py
