import os
import cv2 as cv

# C:\Users\tomas\opencv\opencv\build\x64\vc15\bin\opencv_createsamples.exe -info mschalice.txt -w 24 -h 24 -num 1000 -vec mschalice.vec
# C:\Users\tomas\opencv\opencv\build\x64\vc15\bin\opencv_traincascade.exe -data cascade/ -vec mschalice.vec -bg cagneycarnation.txt -w 24 -h 24 -numPos 85 -numNeg 500 -numStages 10
dirName = os.path.join(os.getcwd(), 'pictures', 'Ms.Chalice')
listOfFiles = list()


for (dirpath, dirnames, filenames) in os.walk(dirName):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]


with open('mschalice.txt', 'w') as f:
    for elem in listOfFiles:
        img = cv.imread(elem)
        dimensions = img.shape
        f.write(elem + f' 1 0 0 {dimensions[1]} {dimensions[0]} \n')
