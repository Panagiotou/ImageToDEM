import sys, os

dem = None
jpg = None
mul = None
if(len(sys.argv) > 1): dem = sys.argv[1]
else:
    print("Please provide arguments:\n\t" + sys.argv[0] + " dem.tif image.jpg (optional) mul (optional, example m5)")
    exit(1)

if(len(sys.argv) > 1):
    if(sys.argv[1].endswith(".jpg")):
        jpg = sys.argv[1]

if(len(sys.argv) > 2):
    a3 = sys.argv[2]
    if("m" in a3):
        mul = int(sys.argv[2][1:])

if(len(sys.argv) > 3): print("Too many arguments!")

if(dem):
    dem = dem.replace(' ', '\ ')
if(jpg):
    jpg = jpg.replace(' ', '\ ')

if(mul):
    os.system("python3  multDEM.py " + dem + " " + str(mul))
    dem = dem[:-4] + '-mul.tif'
    
os.system("cp " + dem + " ../Visualization/threejs-dem-visualizer-master/src/textures/agri-small-dem.tif")
os.system("cp " + jpg + " ../Visualization/threejs-dem-visualizer-master/src/textures/agri-small-autumn.jpg")
os.system("yarn --cwd ../Visualization/threejs-dem-visualizer-master dev")
