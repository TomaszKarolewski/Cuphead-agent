import os
import cv2 as cv


# C:\Users\tomas\opencv\opencv\build\x64\vc15\bin\opencv_createsamples.exe -info mschalice.txt -w 50 -h 50 -num 3000 -vec mschalice.vec
# C:\Users\tomas\opencv\opencv\build\x64\vc15\bin\opencv_traincascade.exe -data cascade/ -vec mschalice.vec -bg cagney.txt -w 50 -h
# 50 -numPos 600 -numNeg 800 -numStages 12 -maxFalseAlarmRate 0.4 -minHitRate 0.999
def create_file_list(pic_file_name, output, spaces_flag=0, position_flag=0):
    dirName = os.path.join(os.getcwd(), 'pictures', pic_file_name)
    listOfFiles = list()

    # removes spaces from directory names
    if spaces_flag == 1:
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
            if position_flag == 1:
                f.write(elem + f' 1 0 0 {dimensions[1]} {dimensions[0]} \n')
            else:
                f.write(elem + '\n')


if __name__ == '__main__':
    create_file_list(pic_file_name='Cagney', output='cagney.txt')
