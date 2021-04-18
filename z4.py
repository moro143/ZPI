import math
import cv2
import scipy.spatial.distance
import numpy as np


img = cv2.imread('home.png')  ## <--  wczytanie zdjęcia 

(rows,cols,_) = img.shape
cv2.imshow('img',img)
dst=0
p = []
d = []
W = 0
H = 0

def xinmin(x, lista, p=0):
    temp = []
    for i in lista:
        temp.append(i[p])
    temp.sort()
    if x in temp[:2]:
        return True
    return False

def disMouse(event, x, y, flags, param):
    cv2.imshow('dst',dst)
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(dst,(x,y),5,(255,255,0),-1)
        d.append((x,y))

def add_p(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),5,(255,255,0),-1)
        p.append((x, y))
        cv2.imshow('img',img)

def calculate(pts):
    temp = [0,0,0,0]
    for i in pts:
        if xinmin(i[0], pts, 0) and xinmin(i[1], pts, 1):
            temp[0] = i
        elif xinmin(i[0], pts, 0) and not xinmin(i[1], pts, 1):
            temp[2] = i
        elif not xinmin(i[0], pts, 0) and xinmin(i[1], pts, 1):
            temp[1] = i
        elif not xinmin(i[0], pts, 0) and not xinmin(i[1], pts, 1):
            temp[3] = i
    pts = temp
    u0 = (cols)/2.0
    v0 = (rows)/2.0

    w1 = scipy.spatial.distance.euclidean(pts[0], pts[1])
    w2 = scipy.spatial.distance.euclidean(pts[2], pts[3])

    h1 = scipy.spatial.distance.euclidean(pts[0], pts[2])
    h2 = scipy.spatial.distance.euclidean(pts[1], pts[3])

    w = max(w1,w2)
    h = max(h1,h2)

    ar_vis = float(w)/float(h)

    m1 = np.array((pts[0][0],pts[0][1],1)).astype('float32')
    m2 = np.array((pts[1][0],pts[1][1],1)).astype('float32')
    m3 = np.array((pts[2][0],pts[2][1],1)).astype('float32')
    m4 = np.array((pts[3][0],pts[3][1],1)).astype('float32')


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

    ar_real = math.sqrt(np.dot(np.dot(np.dot(n2,Ati),Ai),n2)/np.dot(np.dot(np.dot(n3,Ati),Ai),n3))

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    pts1 = np.array(pts).astype('float32')
    pts2 = np.float32([[0,0],[W,0],[0,H],[W,H]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return M, W, H, pts1, pts2
cv2.setMouseCallback('img', add_p)
cv2.waitKey(0)
cv2.destroyAllWindows()


M , W, H, pts1, pts2 = calculate(p)
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img ,M, (W,H))

cv2.imshow('dst',dst)
cv2.setMouseCallback('dst', disMouse)
#cv2.imwrite('orig.png',img)
#cv2.imwrite('proj.png',dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

l = input("Wymiar: ")
result = float(l)/np.sqrt((d[0][0]-d[1][0])**2+(d[0][1]-d[1][1])**2)
roof_height = result*H
roof_width = result*W


from sqrs import numer_sqrs

panel_poziomo = 195.6
panel_pionowo = 99.2
odstep = 2
ip1, ip2 = numer_sqrs((roof_width, roof_height), (panel_poziomo, panel_pionowo), odstep)

img = cv2.imread('home.png')
dst = cv2.warpPerspective(img ,M, (W,H))
x = cv2.imread('BLUE.jpg')

xx, yy = int(panel_pionowo/result), int(panel_poziomo/result)

y = cv2.resize(x, (xx,yy))
print(ip1, ip2)
for i in range(ip2):
    for j in range(ip1):
        dst[i*y.shape[0]:i*y.shape[0]+y.shape[0], j*y.shape[1]:(j+1)*y.shape[1]] = y

cv2.imshow('TEST',dst)
M = cv2.getPerspectiveTransform(pts2,pts1)
test_2 = cv2.warpPerspective(dst , M, (img.shape[1],img.shape[0]))
cv2.resize(img, (img.shape[1],img.shape[0]))
cv2.imshow('test_2', test_2)
cv2.imshow('i',img)
cv2.waitKey(0)

for i in range(len(test_2)):
    for j in range(len(test_2[i])):
        c = 0
        for k in range(len(test_2[i][j])):
            if test_2[i][j][k]==0:
                c+=1
        if c!=3:
            img[i][j]=test_2[i][j]

cv2.imshow('ii', img)
cv2.waitKey(0)
cv2.destroyAllWindows()