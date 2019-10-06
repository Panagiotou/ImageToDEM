import csv
import os
dir = "sample/"
filename = 'sampleor.csv'
percentageUsed = 0.5 #Percentage of csv file lines used(higher accuracy - higher file size)
output = 'sampleReduced'

i = 0
X = []
Y = []
Z = []
with open(dir + filename) as f:
  reader = csv.reader(f)
  for row in reader:
    # do something here with `row`
    if(i%int(1/percentageUsed) == 0):
        if(float(row[2]) > 0):
            X.append(float(row[0]))
            Y.append(float(row[1]))
            Z.append(float(row[2]))
    i+=1

with open(dir + output + '.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(X)):
        filewriter.writerow([str(X[i]), str(Y[i]), str(Z[i])])

os.system("gdal_translate -a_srs EPSG:4326 -of GTiff " + dir + output + ".csv " + dir + output + ".tif")
