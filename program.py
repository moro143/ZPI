import PySimpleGUI as sg
import wymiary
import interface

layout = [
    [sg.Button('Wczytaj zdjęcie')],
    [sg.Button('Podaj wymiary')]
]

window = sg.Window("Panele fotowoltaiczne", layout)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "Wczytaj zdjęcie":
        interface.interface()
    if event == "Podaj wymiary":
        wymiary.wymiary()