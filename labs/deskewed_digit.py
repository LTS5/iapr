# https://stackoverflow.com/questions/51237834/deskewing-mnist-dataset-images-using-minarearect-of-opencv
import os
import numpy as np
import cv2  # 4.5.5


def deskewed_img(src_path, digit, index):
    image_name = "{}/{}_{}.png".format(digit, digit, index)
    print(image_name)
    image_path = os.path.join(src_path, image_name)
    image = cv2.imread(image_path)  # for 4##5032,6780 #8527,2436,1391
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  ## for 9 problem with 4665,8998,73,7

    #%% original
    # gray = cv2.bitwise_not(gray)
    # Gblur = cv2.blur(gray, (5, 5))
    # thresh = cv2.threshold(Gblur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # _, contours = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cnt1 = contours[0]
    # print(cnt1)
    # cnt = cv2.convexHull(contours[0])

    #%% method from https://github.com/rtharungowda/hcg/blob/2cc3125e120d3279bfe8daf89e16f28d38ac4767/segmentation.py
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    # dilate = cv2.dilate(thresh, kernel, iterations=5)
    # contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # cnt = contours[0]

    #%% method2: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    # ensure the contour as coovex hull
    cnt = cv2.convexHull(cnt)

    #%% compute min area rectangle
    rect = cv2.minAreaRect(cnt)
    angle = rect[-1]
    print("Actual angle is", str(angle))
    if cnt.shape[0] > 5:
        ellipse = cv2.fitEllipse(cnt)
        angle_ell = ellipse[-1]
        print("Ellipse axis angle:", angle_ell)

    # angle calculation: https://namkeenman.wordpress.com/2015/12/21/opencv-determine-orientation-of-ellipserotatedrect-in-fitellipse/
    # https://stackoverflow.com/a/62701632
    p = np.array(rect[1])
    # print(p[0])
    if p[0] < p[1]:
        print("Angle along the longer side:", str(rect[-1] + 180))
        act_angle = angle + 180
        # act_angle_ell = angle_ell + 180
        print("Angle along the longer side (ell):", str(angle_ell + 180))
    else:
        print("Angle along the longer side:", str(rect[-1] + 90))
        act_angle = angle + 90
        # act_angle_ell = angle_ell + 90
        print("Angle along the longer side (ell):", str(angle_ell + 90))
    # act_angle gives the angle with bounding box

    if angle_ell < 90:
        angle_ell = angle_ell
    else:
        angle_ell = angle_ell - 180

    if act_angle < 90:
        angle = 90 + angle
        # angle_ell = 90 + angle_ell
        print("angle less than -90")

        # otherwise, just take the inverse of the angle to make
        # it positive
    else:
        angle = act_angle - 180
        # angle_ell = act_angle_ell - 180
        print("angle greater than 90")

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    print(angle, angle_ell)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M_ell = cv2.getRotationMatrix2D(center, angle_ell, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    rotated_ell = cv2.warpAffine(
        image, M_ell, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    box = cv2.boxPoints(rect)
    # print(box)
    box = np.int0(box)
    # print(box)

    contour = cv2.drawContours(thresh, [box], 0, (0, 0, 255), 1)
    # print("contours" + str(p))

    # cv2.imwrite("post/MinAreaRect9.png",contour)
    # cv2.imwrite("post/Input_9.png", image)
    # cv2.imwrite('post/Deskewed_9.png', rotated)

    image_name = "{}/{}_{}.png".format(digit, digit, index)
    # MinAreaRect_image_name = "{}/{}_{}.png".format(digit, digit, index)
    # Deskewed_image_name = "Deskewed{}/{}_{}.png".format(digit, digit, index)

    # os.makedirs(MinAreaRect_dir, exist_ok=True)
    # os.makedirs(Deskewed_dir, exist_ok=True)
    # cv2.imwrite('test_contour.png', contour)
    # cv2.imwrite('test_rotated.png', rotated)
    os.makedirs('./res/', exist_ok=True)
    cv2.imwrite('./res/{}_{}.png'.format(digit, index), image)
    cv2.imwrite('./res/{}_{}_contour.png'.format(digit, index), contour)
    cv2.imwrite('./res/{}_{}_rotated.png'.format(digit, index), rotated)
    cv2.imwrite('./res/{}_{}_rotated_ell.png'.format(digit, index), rotated_ell)


src_path = '/home/he/projects/iapr/data/lab-02-data/part1/'


def naive():
    digit = 0  # 0 | 1
    index = 3  # 0 - 9
    deskewed_img(src_path, digit, index)


def main():
    digit = 0  # 0 | 1
    index = 3  # 0 - 9
    for digit in [0, 1]:
        for index in range(0, 10):
            deskewed_img(src_path, digit, index)


if __name__ == "__main__":
    main()

# ref
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
# https://github.com/rtharungowda/hcg/blob/2cc3125e120d3279bfe8daf89e16f28d38ac4767/segmentation.py
