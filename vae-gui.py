from dash.dependencies import Input, Output
from dash import Dash
import dash_daq as daq
from dash import html
# from jupyter_dash import JupyterDash
from dash import dcc
# import torch
import numpy as np
import plotly.graph_objs as go
import cv2
from PIL import Image
import pickle

file = open('/data/frey_faces_all_latents_matrix.pkl', 'rb')
all_faces = pickle.load(file)
file.close()
pil_img = Image.open('/data/decoder.png')

app = Dash(__name__)
server = app.server

joystick = daq.Joystick(
    id='our-joystick',
    label=" ",
    angle=0,
    size=250,
    style={'float': 'center', 'margin': 'auto'}
)

face = dcc.Graph(id="latent", config={'staticPlot': True},
                 style={'float': 'center', 'margin': 'auto'}
                 )

app.layout = html.Div([
    html.H1('Latent Space',
            style={'position': 'absolute', 'color': '#b5b5b5', 'fontSize': 20, 'margin-left': 160, 'margin-top': 40,
                   'font-family': 'Arial'}),
    html.H1('\u27F5 \u00A0 Expression \u00A0 \u27F6',
            style={'position': 'absolute', 'color': '#636363', 'fontSize': 15, 'margin-left': 150, 'margin-top': 120,
                   'font-family': 'Arial'}),
    html.H1('\u27F5 \u00A0 Pose \u00A0 \u27F6',
            style={'position': 'absolute', 'color': '#636363', 'fontSize': 15, 'margin-left': 100, 'margin-top': 190,
                   'font-family': 'Arial', "transform": "rotate(270deg)"}),
    html.Div(joystick, style={'position': 'absolute', 'margin-left': 100, 'margin-top': 70}),
    html.Div(face, style={'position': 'absolute', 'margin-left': 400, 'margin-top': 0}),
    html.Img(src=pil_img, style={'position': 'absolute', 'margin-left': 370, 'margin-top': 95, 'width': 75}),
    html.H1('Decoder',
            style={'position': 'absolute', 'color': '#b5b5b5', 'fontSize': 20, 'margin-left': 365, 'margin-top': 185,
                   'font-family': 'Arial', "transform": "rotate(270deg)"}),
    html.H1('Image Reconstruction',
            style={'position': 'absolute', 'color': '#b5b5b5', 'fontSize': 20, 'margin-left': 445, 'margin-top': 40,
                   'font-family': 'Arial'}),
],
    className='column'
)


@app.callback(
    Output('latent', 'figure'),
    Input('our-joystick', 'angle'),
    Input('our-joystick', 'force'),
    # Input('trigger', 'n_intervals')
)
def update_output(angle, force):
    if angle == None or force == None:
        X = 0
        Y = 0

    else:
        X = np.multiply(np.cos(np.deg2rad(angle)), force)
        Y = np.multiply(np.sin(np.deg2rad(angle)), force)

    limit = 7
    latent_X = int(X * limit)
    latent_Y = int(Y * limit)
    if latent_X > 7:
        latent_X = 7
    if latent_X < -7:
        latent_X = -7
    if latent_Y > 7:
        latent_Y = 7
    if latent_Y < -7:
        latent_Y = -7
    img = all_faces[latent_X, latent_Y]

    layout = go.Layout(
        # title="My Dash GG",
        height=380,
        width=300,
    )

    figure = go.Figure(
        go.Heatmap(z=cv2.pyrUp(img), colorscale='gray', colorbar=None, showlegend=False, hoverinfo='none'),
        layout=layout)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_traces(showscale=False)
    return figure


if __name__ == '__main__':
    app.run_server(debug=False)

