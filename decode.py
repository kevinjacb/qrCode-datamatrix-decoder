from pyzbar.pyzbar import decode as qrDecode
from pylibdmtx.pylibdmtx import decode as dmDecode
import cv2 as cv
import numpy as np
import os
import time
'''
    In order to use this function first execute:
    pip install pyzbar, pylibdmtx
'''

path = 'test_qr/'

# This is the function you require.


def decodeImage(image) -> str:
    qrInfo = qrDecode(image)
    dmInfo = dmDecode(image)

    if(not len(qrInfo) and not len(dmInfo)):
        return None

    if(not len(qrInfo)):
        box = dmInfo[0][1]
        # cv.imshow("data matrix", drawBoundingBox(image, box))
        return dmInfo[0][0].decode('utf-8')
    else:
        box = qrInfo[0][1]
        # cv.imshow("qr_code", drawBoundingBox(image, box))
        return qrInfo[0][0].decode('utf-8')


# def drawBoundingBox(image, box):
#     res_image = cv.rectangle(
#         image, box, (255, 0, 0), 3)
#     return res_image


# This is just a demo of the above functions.


# img1 = cv.imread(img_path1)  # reads a data matrix image
# img2 = cv.imread(img_path2)  # reads a qr code image

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

for i, file in enumerate(os.listdir(path)):
    img = cv.imread(os.path.join(path, file))
    sharpened = cv.filter2D(img, -1, kernel)
    print(i+1, " file : ", file, " data : ", decodeImage(sharpened))

# kernel = np.array([[0, -1, 0],
#                    [-1, 5, -1],
#                    [0, -1, 0]])
# img = cv.imread('sharpenedshit.jpg')
# # sharpened = cv.filter2D(img, -1, kernel)
# while True:
#     cv.imshow("one", img)
#     if cv.waitKey(10) & 0xFF == 27:
#         break

# print(decodeImage(img))

# print(decodeImage(img1))
# print(decodeImage(img2))
