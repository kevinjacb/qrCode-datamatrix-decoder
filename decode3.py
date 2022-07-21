from pyzbar.pyzbar import decode as qrDecode
from pylibdmtx.pylibdmtx import decode as dmDecode
import cv2 as cv
import numpy as np
import os

path = 'test_qr/'


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


sr = cv.dnn_superres.DnnSuperResImpl_create()
sr.readModel('EDSR_x4.pb')
sr.setModel('edsr', 4)

cropped = []


def detect_qr(image):
    global cropped
    cropped = []
    k1 = np.ones(shape=(3, 3))
    k2 = np.ones(shape=(5, 5))
    sharpen = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = cv.filter2D(img, -1, kernel=sharpen)
    edge = cv.Sobel(img, -1, 1, 1, ksize=5)
    d1 = cv.dilate(edge, kernel=k1, iterations=2)
    e1 = cv.erode(d1, kernel=k1, iterations=5)

    e1 = cv.GaussianBlur(e1, (3, 3), 1, 1)

    d2 = cv.dilate(e1, kernel=k2, iterations=1)
    e2 = cv.erode(d2, kernel=k2, iterations=4)

    e2 = cv.GaussianBlur(e2, (3, 3), 1, 1)

    d3 = cv.dilate(e2, kernel=k2, iterations=3)
    e3 = cv.erode(d3, kernel=k1, iterations=5)

    highest = e3.max()
    # print(highest)
    _, res = cv.threshold(e3, highest-10, 255, cv.THRESH_BINARY)
    # res = cv.dilate(res,kernel=k2,iterations=30)
    contours, heirarchy = cv.findContours(
        res, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cropped = []
    for c in contours:
        M = cv.moments(c)
        cX = int(M['m10']/M['m00'])
        cY = int(M['m01']/M['m00'])
        # print(cX, cY)
        img_size = 100
        startX, endX = cX - img_size, cX + img_size
        startY, endY = cY - img_size, cY + img_size

        startX = startX if startX > 0 else 0
        startY = startY if startY > 0 else 0

        endX = endX if endX < img.shape[1] else img.shape[1]
        endY = endY if endY < img.shape[0] else img.shape[0]

        cropped.append(image[startY:endY, startX:endX])


for i, filename in enumerate(os.listdir(path)):
    # print(filename)
    if not filename.endswith('.jpg'):
        continue
    img = cv.imread(path+filename)
    res = decodeImage(img)
    if(res == None):
        detect_qr(img)
        for cimg in cropped:
            uimg = sr.upsample(cimg)
            res = decodeImage(uimg)
            if(res != None):
                break
    print(f'{i} file: {filename} data: {res}')
