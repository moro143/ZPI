import math
import cv2
import scipy.spatial.distance
import numpy as np
import csv

class Panel:
    def __init__(self, src, height, width, interspace, price, power, name, weight):
        self.src = src
        self.height = height
        self.width = width
        self.interspace = interspace
        self.img = cv2.imread(src)
        self.price = price
        self.power = power
        self.name = name
        self.weight = weight

class Roof:
    def __init__(self, pts1, pts2, W, H, orginal):
        self.pts1 = pts1
        self.pts2 = pts2
        self.width = W
        self.height = H
        self.M = cv2.getPerspectiveTransform(self.pts1, self.pts2)
        self.img = cv2.warpPerspective(orginal, self.M, (self.width, self.height))
        self.points = []
        self.cm_px = None
        self.list_panels_dist = []
        self.roofs_with_panels = []

    def run(self):
        cv2.imshow('roof', self.img)
        cv2.setMouseCallback('roof', self.add_points)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        l = input("Wymiar: ")
        self.cm_px = float(l)/np.sqrt((self.points[0][0]-self.points[1][0])**2+(self.points[0][1]-self.points[1][1])**2)
        

    def add_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.img, (x, y), 5, (255, 255, 0), -1)
            self.points.append((x, y))
            cv2.imshow('roof', self.img)

    def panels_distribution(self, panel, rotate=False):
        panel_width = panel.width/self.cm_px
        panel_height = panel.height/self.cm_px
        if rotate:
            panel_width, panel_height = panel_height, panel_width
        ip1 = math.floor(self.width / (panel_width+panel.interspace))
        ip2 = math.floor(self.height / (panel_height+panel.interspace))
        self.list_panels_dist.append([[panel, ip1, ip2, rotate]])
    
    def panels_distribution_mix(self, panel, rotate = False):
        panel_width = panel.width/self.cm_px
        panel_height = panel.height/self.cm_px
        if rotate:
            panel_width, panel_height = panel_height, panel_width
        ip1 = math.floor(self.width / (panel_width+panel.interspace))
        ip2 = math.floor(self.height / (panel_height+panel.interspace))
        max_panels = ip1*ip2
        max_panels_list = [[panel, ip1, ip2, rotate]]
        for i in range(ip1):
            sw = self.width - i*(panel_width+panel.interspace)
            ip11 = math.floor(sw / (panel_height+panel.interspace))
            ip22 = math.floor(self.height / (panel_width+panel.interspace))
            if ip11*ip22+i*ip2>max_panels:
                max_panels_list = [[panel, i, ip2, rotate], [panel, ip11, ip22, not rotate]]
        self.list_panels_dist.append(max_panels_list)

    def create_roofs_with_panels(self):
        for i in self.list_panels_dist:
            dst = self.img.copy()
            t=0
            t2=0
            test = 0
            test2 = 0
            c = 0
            for j in i:
                panel_img = j[0].img.copy()
                panel_w = j[0].width/self.cm_px
                panel_h = j[0].height/self.cm_px
                if c%2==1:
                    t2 = 0
                else:
                    t = 0
                if j[3] == True:
                    panel_w, panel_h = panel_h, panel_w
                    panel_img = cv2.rotate(panel_img, cv2.cv2.ROTATE_90_CLOCKWISE)
                    
                panel_img = cv2.resize(panel_img, (int(panel_w), int(panel_h)))
                for k in range(j[2]):
                    for l in range(j[1]):
                        s1 = k*panel_img.shape[0]+k*j[0].interspace
                        s2 = l*panel_img.shape[1]+l*j[0].interspace
                        dst[s1+t2:s1+panel_img.shape[0]+t2, s2+t:s2+panel_img.shape[1]+t] = panel_img
                        test = s2+panel_img.shape[1]
                        test2 = s1+panel_img.shape[0]
                t = test
                t2 = test2
                c+=1
            self.roofs_with_panels.append(dst)
    
    def print_info(self):
        for i in self.list_panels_dist:
            sum_price = 0
            sum_power = 0
            sum_weight = 0
            c = 0
            for j in i:
                sum_price += j[2]*j[1]*j[0].price
                sum_power += j[2]*j[1]*j[0].power
                sum_weight += j[2]*j[1]*j[0].weight
                c += j[2]*j[1]
            n = j[0].name
            print(f"Panele: {n}, Ilosc: {c}, Cena: {sum_price} zl, Moc: {sum_power} W, Waga: {sum_weight} kg")


class Photo:
    def __init__(self, src):
        self.src = src
        self.img = cv2.imread(src)
        self.points = []
        (self.rows, self.cols, _) = self.img.shape
        self.roof = None
        self.photo_with_panels = []

    def run(self):
        cv2.imshow(self.src, self.img)
        cv2.setMouseCallback(self.src, self.add_points)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def add_points(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.img, (x,y), 5, (255,255,0) , -1)
            self.points.append((x, y))
            cv2.imshow(self.src, self.img)
    
    def calculate(self):
        temp = [0, 0, 0, 0]
        t = self.points[0]
        for i in self.points:
            if i[1]<t[1]:
                t = i
        temp[0] = t
        for i in self.points:
            if t[0]<i[0]:
                t = i
        temp[1] = t
        for i in self.points:
            if i not in temp and i[0]<t[0]:
                t = i
        temp[2] = t
        for i in self.points:
            if i not in temp:
                t = i
        temp[3] = t
        self.points = temp
        u0 = (self.cols)/2.0
        v0 = (self.rows)/2.0

        w1 = scipy.spatial.distance.euclidean(self.points[0], self.points[1])
        w2 = scipy.spatial.distance.euclidean(self.points[2], self.points[3])

        h1 = scipy.spatial.distance.euclidean(self.points[0], self.points[2])
        h2 = scipy.spatial.distance.euclidean(self.points[1], self.points[3])

        w = max(w1,w2)
        h = max(h1,h2)

        ar_vis = float(w)/float(h)
        
        m1 = np.array((self.points[0][0],self.points[0][1],1)).astype('float32')
        m2 = np.array((self.points[1][0],self.points[1][1],1)).astype('float32')
        m3 = np.array((self.points[2][0],self.points[2][1],1)).astype('float32')
        m4 = np.array((self.points[3][0],self.points[3][1],1)).astype('float32')


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

        pts1 = np.array(self.points).astype('float32')
        pts2 = np.float32([[0,0],[W,0],[0,H],[W,H]])
        self.roof = Roof(pts1, pts2, W, H, self.img)

    def create_photo_with_panels(self):
        for i in self.roof.roofs_with_panels:
            M = cv2.getPerspectiveTransform(self.roof.pts2, self.roof.pts1)
            roof = cv2.warpPerspective(i, M, (self.img.shape[1], self.img.shape[0]))
            img = self.img.copy()
            cv2.resize(img, (img.shape[1], img.shape[0]))
            img = cv2.addWeighted(img, 1, roof, -255, 0)
            img = cv2.addWeighted(img, 1, roof, 1, 0)
            self.photo_with_panels.append(img)

panels = []
with open('data.csv') as data:
    csv_reader = csv.reader(data, delimiter=';')
    c=0
    for i in csv_reader:
        if c!=0:
            panels.append(Panel('BLUE.jpg', int(int(i[2])/10), int(int(i[3])/10), 2, int(i[5]), int(i[4]), i[1], float(i[6])))
        c+=1
panels = panels[1:]

p = Panel('BLUE.jpg', 208, 103, 2, 200, 200, 'Blue', 20)
x = Photo('home.png')
x.run()
x.calculate()
x.roof.run()
for i in panels:
    x.roof.panels_distribution(i)
    x.roof.panels_distribution(i, True)
    #x.roof.panels_distribution_mix(i)
    x.roof.panels_distribution_mix(i, True)
x.roof.print_info()

"""
x.roof.create_roofs_with_panels()
x.create_photo_with_panels()
y=0
for i in x.photo_with_panels:
    cv2.imshow(str(y), i)
    y+=1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""