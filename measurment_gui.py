import PySimpleGUI as sg

def return_measurment():
    layout = [
        [sg.Text("Podaj wymiary[cm]: "), sg.Input()],
        [sg.Button("OK")]
    ]
    window = sg.Window('Pomiar', layout)
    _, values = window.read()
    window.close()
    return int(values[0])
