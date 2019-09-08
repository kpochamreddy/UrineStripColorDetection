import cv2
import numpy as np
from scipy.spatial import distance as dist
squares_pos=[[41,80],[41,198],[41,316],[41,439],[41,556],[41,673],[41,795],[41,915],[41,1030],[41,1155],[41,1270]]
def D(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    maxWidth = 87

    # compute the height of the new image, which will be the
    maxHeight = 1677

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped
img = cv2.imread('1567889240105.JPEG')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
gray=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,100,200)
(_, im_bw) = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow('image',cv2.resize(im_bw,(0,0),fx=0.5,fy=0.5))
# cv2.waitKey(0)
cnt, hierarchy = cv2.findContours(im_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
boxes=[]
for c in cnt:
    are = cv2.contourArea(c)
    if (are < 100 ):
        continue
    peri = cv2.arcLength(c, True)
    if(peri>5000):
        continue
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    if len(approx) !=4:
        continue
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    boxes.append(box)
    # cv2.drawContours(img, [c], 0, (0, 255, 0), 3)
    if(len(boxes)==2):
        break

    # cv2.drawContours(img, [c], 0, (0, 255, 0), 3)
n=len(boxes)
if(n>0):
    if(n==1):
        box=order_points(boxes[0])
    else:
        box1 = order_points(boxes[0])
        box2 = order_points(boxes[1])
        d1=D(box1[0],box1[1])
        d2=D(box1[0],box1[3])
        if(d1>d2):
            if(box1[0][0]<box2[0][0]):
                box=box1
                box[1] = box2[1]
                box[2] = box2[2]
            else:
                box=box2
                box[1] = box1[1]
                box[2] = box1[2]
        else:
            if (box1[0][1] < box2[0][1]):
                box = box1
                box[1] = box2[2]
                box[2] = box2[3]
            else:
                box = box2
                box[1] = box1[2]
                box[2] = box1[3]

    d1 = D(box[0], box[1])
    d2 = D(box[0], box[3])
    wimg=four_point_transform(img,box)
    rgb_values=[]
    w=10
    h=10
    for i in range(len(squares_pos)):
        pos=squares_pos[i]
        x,y=pos
        s_img=np.array(wimg[y-h:y+h,x-w:x+w])
        m_v=np.int0(np.round(s_img.mean(axis=(0,1))))
        rgb_values.append(m_v)
    # cv2.imshow('image',cv2.resize(wimg,(0,0),fx=0.5,fy=0.5))
    # cv2.waitKey(0)
    print(rgb_values)