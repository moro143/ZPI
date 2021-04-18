import math

def numer_sqrs(dach=(10.3, 7), panel=(1.96, 1.63), odstep=0.02):
    bok_dachu_poziomy = dach[0]
    bok_dachu_pionowy = dach[1]
    bok_panelu_poziomy = panel[0] + odstep
    bok_panelu_pionowy = panel[1] + odstep

    ilość_paneli_1 = math.floor(bok_dachu_poziomy / bok_panelu_poziomy)
    ilość_paneli_2 = math.floor(bok_dachu_pionowy / bok_panelu_pionowy)
    ilość_1 = ilość_paneli_1 * ilość_paneli_2

    ilość_paneli_3 = math.floor(bok_dachu_poziomy / bok_panelu_pionowy)
    ilość_paneli_4 = math.floor(bok_dachu_pionowy / bok_panelu_poziomy)
    ilość_2 = ilość_paneli_3 * ilość_paneli_4

    o = []
    a = math.floor(bok_dachu_poziomy / bok_panelu_poziomy)
    r1 = bok_dachu_poziomy%bok_panelu_poziomy

    for i in range(1,a):
        y = r1 + bok_panelu_poziomy*i
        x = bok_dachu_poziomy - y
        p = 0
        g = 0
        if y / bok_panelu_pionowy >= 1:
            p = math.floor(x/bok_panelu_poziomy)*math.floor(bok_dachu_pionowy/bok_panelu_pionowy)
            g = math.floor(y/bok_panelu_pionowy)*math.floor(bok_dachu_pionowy/bok_panelu_poziomy)
            o.append(p+g)

    b = math.floor(bok_dachu_pionowy / bok_panelu_pionowy)
    r2 = bok_dachu_poziomy%bok_panelu_pionowy

    for i in range(1,b):
        y = r2 + bok_panelu_pionowy*i
        x = bok_dachu_poziomy - y
        p = 0
        g = 0
        if y / bok_panelu_poziomy >= 1:
            p = math.floor(x/bok_panelu_pionowy)*math.floor(bok_dachu_pionowy/bok_panelu_poziomy)
            g = math.floor(y/bok_panelu_poziomy)*math.floor(bok_dachu_pionowy/bok_panelu_pionowy)
            o.append(p+g)

    odpowiedz = [(ilość_1,'pionowo'),(ilość_2,'poziomo'),(max(o),'mieszanie')]
    return ilość_paneli_3, ilość_paneli_4

numer_sqrs()