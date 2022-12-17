import cv2 as cv
from time import time
from WindowCapture import WindowCapture
from ObjectFrame import draw_text_on_bg, draw_rectangle

cascade_chalice = cv.CascadeClassifier('cascade/cascade.xml')

wincap = WindowCapture()
loop_time = time()
while True:
    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    gray = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
    rectangle = cascade_chalice.detectMultiScale(gray, minNeighbors=5, minSize=(90,90))
    for (x,y,w,h) in rectangle:
        draw_rectangle(screenshot, 'Ms. Chalice', position=(x, y), shape=(w, h), rect_color=(69,195,249))

    # debug the loop rate
    draw_text_on_bg(screenshot, text='FPS: {:.0f}'.format(1 / (time() - loop_time)), position=(0, 0))
    cv.imshow('Computer Vision', screenshot)
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break
