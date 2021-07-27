import PySimpleGUI as sg
import os.path
import z4
from csv import reader
import visual
import traceback

def interface():
    file_list_column = [
        [
            sg.Text("Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse()
        ],
        [
            sg.Listbox(
                values=[], enable_events=True,size=(60,20), key="-FILE LIST-"
            )
        ]
    ]

    image_viewer_column = [ 
        [sg.Button("Projektuj")]
    ]

    layout = [
        [
            sg.Column(file_list_column),
            sg.Column(image_viewer_column)
        ]
    ]
    window = sg.Window("Wybor zdjecia", layout)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                file_list = os.listdir(folder)
            except:
                file_list = []
            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith((".png",".jpg", ".jpeg"))
            ]
            window["-FILE LIST-"].update(fnames)


        if event == "Projektuj":
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                panels = []
                with open('data.csv') as data:
                    csv_reader = reader(data, delimiter=';')
                    c=0
                    for i in csv_reader:
                        print(i)
                        if c!=0  and len(i)>=6:
                            panels.append(z4.Panel('BLUE.jpg', int(int(i[2])/10), int(int(i[3])/10), 2, int(i[5]), int(i[4]), i[1], float(i[6])))
                        c+=1
                panels = panels[1:]
                print(filename)
                x = z4.Photo(values["-FILE LIST-"][0])
                #x = z4.Photo(filename)
                x.run()
                x.calculate()
                x.roof.run()
                for i in panels:
                    x.roof.panels_distribution(i)
                    x.roof.panels_distribution(i, True)
                    x.roof.panels_distribution_mix(i, True)
                x.roof.create_roofs_with_panels()
                x.create_photo_with_panels()
                visual.visualizations(x)
                
            except Exception as inst:
                print(inst)
                traceback.print_exc()
                pass

#interface()