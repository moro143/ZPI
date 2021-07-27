import PySimpleGUI as sg
import visual
import z4
from csv import reader
import numpy as np
import cv2

def wymiary():
    layout = [
        [sg.Text("Podaj szerokość[cm]: "), sg.Input()],
        [sg.Text("Podaj wysokość[cm]: "), sg.Input()],
        [sg.Button("Gotowe")]
    ]

    window = sg.Window('Wymiary dachu', layout)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Gotowe":
            W = int(values[0])
            H = int(values[1])
            panels = []
            with open('data.csv') as data:
                csv_reader = reader(data, delimiter=';')
                c=0
                for i in csv_reader:
                    if c!=0:
                        panels.append(z4.Panel('BLUE.jpg', int(int(i[2])/10), int(int(i[3])/10), 2, int(i[5]), int(i[4]), i[1], float(i[6])))
                    c+=1
            panels = panels[1:]
            
            x = z4.Photo('BLUE.jpg')
            x.points = [[0,0],[H,W],[H,0],[0,W]]
            x.roof = z4.Roof(W=W,H=H, orginal=x.img, visualization=False)
            x.roof.cm_px = 1
            x.roof.list_panels_dist = []
            x.roof.roofs_with_panels = []
            x.roof.img = x.img

            for i in panels:
                x.roof.panels_distribution(i)
                x.roof.panels_distribution(i, True)
                x.roof.panels_distribution_mix(i, True)
            visual.visualizations(x, False)
            