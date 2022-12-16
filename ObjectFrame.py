import cv2 as cv


def draw_text_on_bg(img, text, position=(0, 0), font=cv.FONT_HERSHEY_PLAIN, font_scale=1, font_thickness=1,
                    text_color=(0, 0, 0), background_color=(255, 255, 255)):
    """
    function draws text with background on a given image
    :param img: image to draw text on
    :param text: text to pass on
    :param position: top left origin of the text box
    :param font:
    :param font_scale: font size
    :param font_thickness:
    :param text_color: RGB tuple
    :param background_color: RGB tuple
    """
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    cv.rectangle(img, position, (position[0] + text_size[0], position[1] + text_size[1] + 2), background_color, -1)
    cv.putText(img, text, (position[0], position[1] + text_size[1] + font_scale - 1), font, font_scale, text_color,
               font_thickness)


def draw_rectangle(img, text, position=(0, 0), shape=(200, 200), rect_color=(255, 255, 255), rect_thickness=2,
                   font=cv.FONT_HERSHEY_PLAIN, font_scale=1, font_thickness=1, text_color=(255, 255, 255)):
    """
    function draws rectangle on a given image with additional nametag
    :param img: image to draw rectangle on
    :param text: text to pass on nametag
    :param position: top left origin of the rectangle
    :param shape: rectangle size
    :param rect_color: rectangle color, RGB tuple
    :param rect_thickness: rectangle thickness
    :param font: nametag font
    :param font_scale: nametag font size
    :param font_thickness: nametag font thickness
    :param text_color: nametag text color, RBG tuple
    """
    cv.rectangle(img, position, (position[0] + shape[0], position[1] + shape[1]), rect_color, rect_thickness)
    draw_text_on_bg(img, text, position, font, font_scale, font_thickness, text_color, rect_color)


if __name__ == '__main__':
    image = cv.imread('pictures/test.jpg')
    draw_text_on_bg(image, "FPS: 60", position=(100, 150), font_scale=1)
    cv.imshow('test', image)

    cv.waitKey(0)

if __name__ == '__main__':
    image = cv.imread('pictures/test.jpg')
    draw_rectangle(image, "cuphead", position=(100, 150), rect_color=(0, 0, 255))
    cv.imshow('test', image)

    cv.waitKey(0)
