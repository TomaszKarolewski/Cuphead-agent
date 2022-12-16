import os
import cv2 as cv


# C:\Users\tomas\opencv\opencv\build\x64\vc15\bin\opencv_createsamples.exe -info mschalice.txt -w 24 -h 24 -num 1000 -vec mschalice.vec
# C:\Users\tomas\opencv\opencv\build\x64\vc15\bin\opencv_traincascade.exe -data cascade/ -vec mschalice.vec -bg cagney.txt -w 24 -h 24 -numPos 85 -numNeg 500 -numStages 10
def create_file_list(pic_file_name, output):
    dirName = os.path.join(os.getcwd(), 'pictures', pic_file_name)
    listOfFiles = list()

    # removes spaces from directory names
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        for f in dirnames:
            r = f.replace(" ", "_")
            if r != f:
                os.rename(os.path.join(dirpath, f), os.path.join(dirpath, r))

    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    with open(output, 'w') as f:
        for elem in listOfFiles:
            img = cv.imread(elem)
            dimensions = img.shape
            f.write(elem + f' 1 0 0 {dimensions[1]} {dimensions[0]} \n')


if __name__ == '__main__':
    create_file_list(pic_file_name='Cagney', output='cagney.txt')
