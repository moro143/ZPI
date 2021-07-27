import PySimpleGUI as sg
from cv2 import imshow, waitKey, destroyAllWindows

def max_l(l, name):
    c = l[0][name]
    idx = 0
    for i in range(len(l)):
        if c<l[i][name]:
            c = l[i][name]
            idx = i
    return idx

def min_l(l, name):
    c = l[0][name]
    idx = 0
    for i in range(len(l)):
        if c>l[i][name]:
            c = l[i][name]
            idx = i
    return idx

def get_list(l, name, idx):
    result = []
    for i in range(len(l)):
        if l[idx][name]==l[i][name]:
            result.append(i)
    return result

def color_chosen(chosen, n, listbox):
    for i in chosen:
        listbox.itemconfigure(n[i], bg='green', fg='white')

def get_min_price(l, name='price'):
    c = l[0][name]
    for i in l:
        if i[name]<c:
            c=i[name]
    return c
def get_max_price(l, name='price'):
    c = l[0][name]
    for i in l:
        if i[name]>c:
            c=i[name]
    return c


def visualizations(x, wizualizacja = True):
    tmp = x.roof.info()
    l = []
    n = {}
    nn = {}
    t = 0
    chosen = []
    for i in tmp:
        l.append(i['name']+" "+i['rotation'])
        n[i['name']+" "+i['rotation']] = t
        nn[i['name']+" "+i['rotation']] = t
        t+=1
    sortowanie = ['Najniższa cena', 'Największa moc', 'Najmniejsza waga', 'Najtańsza cena za 1W']
    layout = [
                [sg.Text('Sortowanie')],
                [sg.Listbox(values=sortowanie, enable_events=True, size=(40,len(sortowanie)), key="-SORT-")],
                [sg.Text('Maksymalna cena', size=(20,1)), sg.Slider(range=(get_min_price(tmp), get_max_price(tmp)), orientation='h', size=(34, 20), default_value=get_max_price(tmp), key='-SLIDER PRICE-', enable_events=True)],
                [sg.Text('Maksymalna waga', size=(20,1)), sg.Slider(range=(get_min_price(tmp, 'weight'), get_max_price(tmp, 'weight')), orientation='h', size=(34, 20), default_value=get_max_price(tmp, 'weight'), key='-SLIDER WEIGHT-', enable_events=True)],
                [sg.Text('Minimalna moc', size=(20,1)), sg.Slider(range=(get_min_price(tmp, name='power'), get_max_price(tmp, name='power')), orientation='h', size=(34,20), default_value=get_min_price(tmp, name='power'), key='-SLIDER POWER-', enable_events=True)],
                [sg.Text(size=(60,1), key="-INFO-")],
                [sg.Listbox(values=l, enable_events=True, size=(60,20), key="-LIST-")]#, [sg.Button("Zaznacz/Odznacz"), sg.Button("Odznacz wszystkie")]]
                
            ]
    if wizualizacja:
        layout[-1].append(sg.Button("Wizualizacja"))
    window = sg.Window('Wizualizacje', layout, finalize=True)
    listbox = window['-LIST-'].Widget

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Wizualizacja":
            try:
                imshow("Przyklad",x.photo_with_panels[nn[values["-LIST-"][0]]])
                waitKey(0)
                destroyAllWindows()
            except:
                pass
        if event == "-SORT-":
            idxs=[]
            if values['-SORT-'][0]=='Najniższa cena':
                temp = []
                for i in tmp:
                    temp.append(i['price'])
                idxs = sorted(range(len(temp)), key=lambda k: temp[k])
            if values['-SORT-'][0]=='Najtańsza cena za 1W':
                temp = []
                for i in tmp:
                    temp.append(i['stosunek'])
                idxs = sorted(range(len(temp)), key=lambda k: temp[k])
            if values['-SORT-'][0]=='Najmniejsza waga':
                temp = []
                for i in tmp:
                    temp.append(i['weight'])
                idxs = sorted(range(len(temp)), key=lambda k: temp[k])
            if values['-SORT-'][0]=='Największa moc':
                temp = []
                for i in tmp:
                    temp.append(i['power'])
                idxs = sorted(range(len(temp)), key=lambda k: temp[k], reverse=True)
            
            if idxs!=[]:
                tl = []
                ttmp = []
                for i in idxs:
                    tl.append(l[i])
                    ttmp.append(tmp[i])
                l = tl
                t = 0
                for i in ttmp:
                    n[i['name']+" "+i['rotation']] = t
                    t+=1
                tmp = ttmp
                window.Element('-LIST-').Update(values=l)
                for i in range(len(tmp)):
                    listbox.itemconfigure(i, bg='white', fg='black')
                color_chosen(chosen, n, listbox)
            tmpl = []
            for i in tmp:
                if i['price']<=values['-SLIDER PRICE-']:
                    if i['weight']<=values['-SLIDER WEIGHT-']:
                        if i['power']>=values['-SLIDER POWER-']:
                            tmpl.append(i['name']+' '+i['rotation'])
            window.Element('-LIST-').Update(values=tmpl)
        if event == 'Zaznacz/Odznacz':
            try:
                if values['-LIST-'][0] in chosen:
                    chosen.remove(values['-LIST-'][0])
                else:
                    chosen.append(values['-LIST-'][0])
                for i in range(len(tmp)):
                    listbox.itemconfigure(i, bg='white', fg='black')
                color_chosen(chosen, n, listbox)
            except:
                pass
        if event == "Odznacz wszystkie":
            try:
                chosen=[]
                for i in range(len(tmp)):
                    listbox.itemconfigure(i, bg='white', fg='black')
                color_chosen(chosen, n, listbox)
            except:
                pass
        if event == "-LIST-":
            window["-INFO-"].update('Ilosc: '+ str(tmp[n[values["-LIST-"][0]]]['quantity']) +' Waga: '+str(tmp[n[values["-LIST-"][0]]]['weight'])+' Moc: '+str(tmp[n[values["-LIST-"][0]]]['power'])+' Cena: '+str(tmp[n[values["-LIST-"][0]]]['price'])+' Cena 1W: '+str(tmp[n[values["-LIST-"][0]]]['stosunek']))
        if event == '-SLIDER PRICE-' or event=='-SLIDER WEIGHT-' or event == '-SLIDER POWER-':
            tmpl = []
            for i in tmp:
                if i['price']<=values['-SLIDER PRICE-']:
                    if i['weight']<=values['-SLIDER WEIGHT-']:
                        if i['power']>=values['-SLIDER POWER-']:
                            tmpl.append(i['name']+' '+i['rotation'])
            window.Element('-LIST-').Update(values=tmpl)
    window.close()
    return 0