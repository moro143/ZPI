import math
import cv2
import scipy.spatial.distance
import numpy as np

panel_pionowo = 168.9
panel_poziomo = 99.6
odstep = 2
home = 'home.png'
#panel = 'BLUE.jpg'
#panel = 'ja.png'
panel = 'eco.png'
#panel = 'a.jfif'

img = cv2.imread(home)  ## <--  wczytanie zdjęcia 

(rows,cols,_) = img.shape
cv2.imshow('img',img)
dst=0
p = []
d = []
W = 0
H = 0

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
    print(pts)
    t = (10000,10000)
    for i in pts:
        if i[1]<t[1]:
            t = i
    temp[0] = t
    for i in pts:
        if t[0]<i[0]:
            t = i
    temp[1] = t
    for i in pts:
        if i not in temp and i[0]<t[0]:
            t = i
    temp[2] = t
    for i in pts:
        if i not in temp:
            t=i
    temp[3] = t
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
    print(pts)
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



import math

ip1 = math.floor(roof_width / (panel_poziomo+odstep))
ip2 = math.floor(roof_height / (panel_pionowo + odstep))

img = cv2.imread(home)
dst = cv2.warpPerspective(img ,M, (W,H))
x = cv2.imread(panel)

xx, yy = int(panel_poziomo/result), int(panel_pionowo/result)

y = cv2.resize(x, (xx,yy))

for i in range(ip2):
    for j in range(ip1):
        dst[i*y.shape[0]+i*odstep:i*y.shape[0]+y.shape[0]+i*odstep, j*y.shape[1]+j*odstep:(j+1)*y.shape[1]+j*odstep] = y

M1 = cv2.getPerspectiveTransform(pts2,pts1)
test_2 = cv2.warpPerspective(dst , M1, (img.shape[1],img.shape[0]))
cv2.resize(img, (img.shape[1],img.shape[0]))

test_3 = cv2.addWeighted(img,1,test_2,-255,0)
test_4 = cv2.addWeighted(test_3,1, test_2,1,0)
cv2.imshow('test_4', test_4)

################################################################################################################################

panel_pionowo, panel_poziomo = panel_poziomo, panel_pionowo
ip1 = math.floor(roof_width / (panel_poziomo+odstep))
ip2 = math.floor(roof_height / (panel_pionowo+odstep))

img = cv2.imread(home)
dst = cv2.warpPerspective(img ,M, (W,H))
x = cv2.imread(panel)
x = cv2.rotate(x, cv2.cv2.ROTATE_90_CLOCKWISE)

xx, yy = int(panel_poziomo/result), int(panel_pionowo/result)

y = cv2.resize(x, (xx,yy))
for i in range(ip2):
    for j in range(ip1):
        dst[i*y.shape[0]+i*odstep:i*y.shape[0]+y.shape[0]+i*odstep, j*y.shape[1]+j*odstep:(j+1)*y.shape[1]+j*odstep] = y

M1 = cv2.getPerspectiveTransform(pts2,pts1)
test_5 = cv2.warpPerspective(dst , M1, (img.shape[1],img.shape[0]))
cv2.resize(img, (img.shape[1],img.shape[0]))

test_6 = cv2.addWeighted(img,1,test_5,-255,0)
test_7 = cv2.addWeighted(test_6,1, test_5,1,0)
cv2.imshow('test_7', test_7)

cv2.waitKey(0)
cv2.destroyAllWindows()