import math
import cv2
import scipy.spatial.distance
from csv import reader
import measurment_gui
from numpy import sqrt, array, dot, cross, abs, transpose, linalg, float32

SIZE = 800
PHOTO = "home.png"
wizualizacja_wszystkich = False     #45 wizualizacji
wizualizacja_przyklad = True        #6 pierwszych wizualizacje

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
    def __init__(self, pts1=0, pts2=0, W=0, H=0, orginal=0, visualization=True):
        self.pts1 = pts1
        self.pts2 = pts2
        self.width = W
        self.height = H
        if visualization:
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
        l = measurment_gui.return_measurment()
        self.cm_px = float(l)/sqrt((self.points[0][0]-self.points[1][0])**2+(self.points[0][1]-self.points[1][1])**2)
        

    def add_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.img, (x, y), 5, (255, 255, 0), -1)
            self.points.append((x, y))
            cv2.imshow('roof', self.img)
            if len(self.points)==2:
                cv2.destroyAllWindows()
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points)>=2:
            cv2.destroyAllWindows()

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
        l = []
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
            p = f"Panele: {n}, Ilosc: {c}, Cena: {sum_price} zl, Moc: {sum_power} W, Waga: {sum_weight} kg"
            l.append(p)
        return l
    
    def info(self):
        l = []
        tmp = 0
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
            if tmp==2:
                rotation = "MIX"
                tmp=-1
            elif tmp==1:
                rotation = "POZIOMO"
            else:
                rotation = "PIONOWO"
            tmp+=1
            if {'name':n, 'rotation':rotation, 'quantity': c, 'power': sum_power, 'weight': sum_weight, 'price':sum_price} in l:
                rotation = 'MIX'
            if c>0:
                l.append({'name':n, 'rotation':rotation, 'quantity': c, 'power': round(sum_power,2), 'weight': round(sum_weight, 2), 'price': round(sum_price,2), 'stosunek': round(sum_price/sum_power,2)})
            else:
                l.append({'name':n, 'rotation':rotation, 'quantity': c, 'power': round(sum_power,2), 'weight': round(sum_weight, 2), 'price': round(sum_price,2), 'stosunek': 0})

        return l


class Photo:
    def __init__(self, src):
        self.src = src
        self.img = cv2.imread(src)
        r = SIZE
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        print(rows, cols)
        c = int(r/rows*cols)
        self.img = cv2.resize(self.img, (c, r))
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
            if len(self.points)==4:
                cv2.destroyAllWindows()
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points)>=4:
            cv2.destroyAllWindows()
    
    def calculate(self):
        l = self.points
        mlat = sum(x[0] for x in l) / len(l)
        mlng = sum(x[1] for x in l) / len(l)
        def algo(x):
            return (math.atan2(x[0] - mlat, x[1] - mlng) + 2 * math.pi) % (2*math.pi)

        l.sort(key=algo)
        self.points = [l[2], l[1], l[3], l[0]]
        u0 = (self.cols)/2.0
        v0 = (self.rows)/2.0

        w1 = scipy.spatial.distance.euclidean(self.points[0], self.points[1])
        w2 = scipy.spatial.distance.euclidean(self.points[2], self.points[3])

        h1 = scipy.spatial.distance.euclidean(self.points[0], self.points[2])
        h2 = scipy.spatial.distance.euclidean(self.points[1], self.points[3])

        w = max(w1,w2)
        h = max(h1,h2)
        ar_vis = float(w)/float(h)
        
        m1 = array((self.points[0][0],self.points[0][1],1)).astype('float32')
        m2 = array((self.points[1][0],self.points[1][1],1)).astype('float32')
        m3 = array((self.points[2][0],self.points[2][1],1)).astype('float32')
        m4 = array((self.points[3][0],self.points[3][1],1)).astype('float32')


        k2 = dot(cross(m1,m4),m3) / dot(cross(m2,m4),m3) #11
        k3 = dot(cross(m1,m4),m2) / dot(cross(m3,m4),m2) #12

        n2 = k2 * m2 - m1 #14
        n3 = k3 * m3 - m1 #15

        n21 = n2[0]
        n22 = n2[1]
        n23 = n2[2]

        n31 = n3[0]
        n32 = n3[1]
        n33 = n3[2]

        f = math.sqrt(abs( (1.0/(n23*n33)) * ((n21*n31 - (n21*n33 + n23*n31)*u0 + n23*n33*u0*u0) + (n22*n32 - (n22*n33+n23*n32)*v0 + n23*n33*v0*v0)))) #31

        A = array([[f,0,u0],[0,f,v0],[0,0,1]]).astype('float32')

        At = transpose(A)
        Ati = linalg.inv(At)
        Ai = linalg.inv(A)

        ar_real = math.sqrt(dot(dot(dot(n2,Ati),Ai),n2)/dot(dot(dot(n3,Ati),Ai),n3))
        if ar_real < ar_vis:
            W = int(w)
            H = int(W / ar_real)
        else:
            H = int(h)
            W = int(ar_real * H)

        pts1 = array(self.points).astype('float32')
        pts2 = float32([[0,0],[W,0],[0,H],[W,H]])
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

if __name__ == "__main__":
    panels = []
    with open('data.csv') as data:
        csv_reader = reader(data, delimiter=';')
        c=0
        for i in csv_reader:
            if c!=0:
                panels.append(Panel('BLUE.jpg', int(int(i[2])/10), int(int(i[3])/10), 2, int(i[5]), int(i[4]), i[1], float(i[6])))
            c+=1
    panels = panels[1:]
    x = Photo(PHOTO)
    x.run()
    x.calculate()
    x.roof.run()
    for i in panels:
        x.roof.panels_distribution(i)
        x.roof.panels_distribution(i, True)
        x.roof.panels_distribution_mix(i, True)
    x.roof.print_info()

    x.roof.create_roofs_with_panels()
    x.create_photo_with_panels()
    y=0
    if wizualizacja_wszystkich:
        
        for i in x.photo_with_panels:
            cv2.imshow(str(y), i)
            y+=1
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif wizualizacja_przyklad:
        for i in x.photo_with_panels:
            cv2.imshow(str(y), i)
            y+=1
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if y==6:
                break
