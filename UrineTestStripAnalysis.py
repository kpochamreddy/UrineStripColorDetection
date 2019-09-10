# needed packages(you need to the following pckages)
import cv2
import numpy as np
from scipy.spatial import distance as dist
# center point of each square for 11 squares
squares_pos=[[41,80],[41,198],[41,316],[41,439],[41,556],[41,673],[41,795],[41,915],[41,1030],[41,1155],[41,1270]]
match_pos=[[[35,20],[87,17],[135,14],[185,17],[235,18]],
           [[22,24],[74,22],[122,21],[172,20]],
           [[26,23],[78,21],[127,20],[177,19],[226,19]],
           [[28,21],[77,20],[128,18],[178,18]],
           [[29,24],[78,23],[128,21],[179,22],[229,22]],
           [[28,26],[77,25],[127,25],[178,25]],
           [[32,57],[387,59],[569,59]],
           [[45,53],[134,53],[223,53],[317,53]],
           [[31,24],[84,28],[135,31],[185,34],[236,37]],
           [[30,22],[82,27],[135,32],[184,38],[233,41],[284,45],[335,47]]]
match_colors=[]
ww=10;hh=10
for i in range(10):
    img_path='Chart/'+str(i+2)+'.JPG'
    img=cv2.imread(img_path)
    colors=[]
    for j in range(len(match_pos[i])):
        x,y=match_pos[i][j]
        img1=img[y-hh:y+hh,x-ww:x+ww]
        m_v = np.int0(np.round(img1.mean(axis=(0, 1))))
        colors.append(m_v)
    match_colors.append(colors)


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
# input image
img = cv2.imread('1567889240105.JPEG')
# remove noise from input image
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
# convert BGR into gray image
gray=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
#  convert gray image into binary image by OTSU method
(_, im_bw) = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow('image',cv2.resize(im_bw,(0,0),fx=0.5,fy=0.5))
# cv2.waitKey(0)
# get the contour of square bar 
cnt, hierarchy = cv2.findContours(im_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# find the square bar box position
boxes=[]
for c in cnt:
    # area of each contour
    # remove small contours 
    are = cv2.contourArea(c)
    if (are < 100 ):
        continue
    # angle of each line in contour
    peri = cv2.arcLength(c, True)
    # remove contour which has many lines
    if(peri>5000):
        continue
    # check whtether contour is rectangle or not. If contour is not rectangle, we remove that contour
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    if len(approx) !=4:
        continue
    # get the rectangle points of contour 
    rect = cv2.minAreaRect(c)
    # get box points from rectangle 
    box = cv2.boxPoints(rect)
    # convert float points into integer points
    box = np.int0(box)
    # add box points into variable boxes
    boxes.append(box)
    cv2.drawContours(img, [c], 0, (0, 255, 0), 3)
    if(len(boxes)==2):
        break
#Main strip contour
#cv2.imshow('image',cv2.resize(img,(0,0),fx=0.5,fy=0.5))
#cv2.waitKey(0)
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
    #Main contour dropped
    wimg=four_point_transform(img,box)
    #cv2.imshow('wimg',wimg)
    #cv2.imwrite('temp.jpg',wimg)
    #cv2.waitKey(0)
    rgb_values=[]
    w=10
    h=10
    match_results=[]
    for i in range(len(squares_pos)):
        pos=squares_pos[i]
        x,y=pos
        s_img=np.array(wimg[y-h:y+h,x-w:x+w])
        m_v=np.int0(np.round(s_img.mean(axis=(0,1))))
        rgb_values.append(m_v)
        if(i==0):
            continue
        # it is minimum differece between square average BGR color and matching average BGR color
        # intial minimum value  .
        M=10000000000 
        index=-1
        m_v=m_v-20
        for j in range(len(match_colors[i-1])):
            m_s=match_colors[i-1][j]
            # difference between square average BGR color and matching average BGR color for each squre.
            # we select match color which has the smallest difference between square average BGR color and matching average BGR color
            d=np.sqrt((m_v[0]-m_s[0])**2+(m_v[1]-m_s[1])**2+(m_v[2]-m_s[2])**2)
            if(M>d):
                M=d
                index=j
        match_results.append(index)


    # cv2.imshow('image',cv2.resize(wimg,(0,0),fx=0.5,fy=0.5))
    # cv2.waitKey(0)
    print(rgb_values)
    print(match_results)