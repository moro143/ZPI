import math
import cv2
import scipy.spatial.distance
import numpy as np

img = cv2.imread('home.png')
(rows,cols,_) = img.shape
cv2.imshow('img',img)
p = []
d = []
W = 0
H = 0
def disMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(d)>=2:
        l = input("Podaj: ")
        result = np.sqrt((d[0][0]-d[1][0])**2+(d[0][1]-d[1][1])**2)/float(l)
        print(result*H)
        print(result*W)
    elif event == cv2.EVENT_LBUTTONDOWN:
        d.append((x,y))

def onMouse(event, x, y, flags, param):
   global W, H
   if len(p)>=4 and event == cv2.EVENT_LBUTTONDOWN:
        #image center
        print(p)
        p.sort()
        p[0],p[1],p[2],p[3] = p[1], p[3], p[0], p[2]
        print(p)
        u0 = (cols)/2.0
        v0 = (rows)/2.0

        #widths and heights of the projected image
        w1 = scipy.spatial.distance.euclidean(p[0],p[1])
        w2 = scipy.spatial.distance.euclidean(p[2],p[3])

        h1 = scipy.spatial.distance.euclidean(p[0],p[2])
        h2 = scipy.spatial.distance.euclidean(p[1],p[3])

        w = max(w1,w2)
        h = max(h1,h2)
        #visible aspect ratio
        ar_vis = float(w)/float(h)

        #make numpy arrays and append 1 for linear algebra
        m1 = np.array((p[0][0],p[0][1],1)).astype('float32')
        m2 = np.array((p[1][0],p[1][1],1)).astype('float32')
        m3 = np.array((p[2][0],p[2][1],1)).astype('float32')
        m4 = np.array((p[3][0],p[3][1],1)).astype('float32')

        #calculate the focal disrance
        k2 = np.dot(np.cross(m1,m4),m3) / np.dot(np.cross(m2,m4),m3) #11
        k3 = np.dot(np.cross(m1,m4),m2) / np.dot(np.cross(m3,m4),m2) #12

        n2 = k2 * m2 - m1 #14
        n3 = k3 * m3 - m1 #15

        n21 = n2[0]
        n22 = n2[1]
        n23 = n2[2]

        n31 = n3[0]
        n32 = n3[1]
        n33 = n3[2]

        f = math.sqrt(np.abs( (1.0/(n23*n33)) * ((n21*n31 - (n21*n33 + n23*n31)*u0 + n23*n33*u0*u0) + (n22*n32 - (n22*n33+n23*n32)*v0 + n23*n33*v0*v0)))) #31

        A = np.array([[f,0,u0],[0,f,v0],[0,0,1]]).astype('float32')

        At = np.transpose(A)
        Ati = np.linalg.inv(At)
        Ai = np.linalg.inv(A)

        #calculate the real aspect ratio
        ar_real = math.sqrt(np.dot(np.dot(np.dot(n2,Ati),Ai),n2)/np.dot(np.dot(np.dot(n3,Ati),Ai),n3))

        if ar_real < ar_vis:
            W = int(w)
            H = int(W / ar_real)
        else:
            H = int(h)
            W = int(ar_real * H)

        pts1 = np.array(p).astype('float32')
        pts2 = np.float32([[0,0],[W,0],[0,H],[W,H]])
        print(W,H)
        #project the image with the new w/h
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(W,H))
        cv2.imshow('dst',dst)

        cv2.setMouseCallback('dst', disMouse)        

        cv2.imwrite('orig.png',img)
        cv2.imwrite('proj.png',dst)
   elif event == cv2.EVENT_LBUTTONDOWN:
        p.append((x, y))

    
cv2.setMouseCallback('img', onMouse)
cv2.waitKey(0)