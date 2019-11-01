import sys, os

dem = None
jpg = None
scale = None
if(len(sys.argv) > 1): dem = sys.argv[1]
else:
    print("Please provide arguments:\n\t" + sys.argv[0] + " dem.tif image.jpg (optional) scale (optional)")
    exit(1)

if(len(sys.argv) > 2):
    if(sys.argv[2].endswith(".jpg")):
        jpg = sys.argv[2]
    else:
        scale = scale = int(sys.argv[2])
if(len(sys.argv) > 3): scale = int(sys.argv[3])
if(len(sys.argv) > 4): print("Too many arguments!")

if(jpg):
    os.system("cp " + dem + " ../Visualization/threejs-dem-visualizer-master/src/textures/agri-small-dem.tif")
    os.system("cp " + jpg + " ../Visualization/threejs-dem-visualizer-master/src/textures/agri-small-autumn.jpg")
    os.system("yarn --cwd ../Visualization/threejs-dem-visualizer-master dev")
else:
    if(scale):
        os.system("python3 DEM2rgb.py " + dem + " " + str(scale))
    else:
        os.system("python3 DEM2rgb.py  " + dem)
