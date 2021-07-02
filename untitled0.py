# img_viewer.py

import PySimpleGUI as sg
import os.path
import cv2
from torch import Tensor
import diagnosis_of_autism

# First the window layout in 2 columns
import torch

model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
#model = models.MobileNetV2()
net = diagnosis_of_autism.Model(model)
PATH = 'C:/Users/Bharathraj C L/Downloads/447704_1090147_bundle_archive/autism/autism_19.pt'

model = diagnosis_of_autism.read_model(PATH,net)

file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout)




# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif",".jpg"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            print(filename,'1')
            
            
            image_data = cv2.imread(filename)
            print(image_data.shape)
            result = diagnosis_of_autism.predict_class(model,image_data)
            print(result,'2')
            _,result = torch.max(result, 1)
            print(result,'3')
            if(result[0] == 0):
                text = 'Child having Autism Disease'
            else:
                text = 'Child not having Autism Disease'
            window["-TOUT-"].update(text)
            imgbytes = cv2.imencode(".png", image_data)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)

            #window["-IMAGE-"].update(data=file_data)

        except:
            pass

window.close()