import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
from skimage import io
import json
from PIL import Image
import shape_utils
import os
import numpy as np
import pandas as pd
import cv2
from netdissect import imgviz
from compute_unit_stats import (
    load_model, load_dataset,
    compute_rq, compute_act, compute_act_quantile,
    cluster_units
)
from torchvision import transforms
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from netdissect.easydict import EasyDict
import plotly.io as pio

pio.templates.default = "simple_white"

external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/image_annotation_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

exp = 'adam_20220315041434'
ckpt_dir = f'../ckpt/vgg16_bn_{exp}'
data_res = 512
num_units = 512
default_layer = 'features.conv5_3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(exp=os.path.basename(ckpt_dir), device=device)
model.retain_layer(default_layer)

# decalare model settings
quantile = 0.99
args = EasyDict(model='vgg16_bn', exp=exp, quantile=quantile)
resdir = 'results/%s-%s-%s-%s' % (args.model, args.exp, default_layer, int(args.quantile * 100))

# iou threshold
iou_th = 0.2

# load dataset
transform = transforms.Compose([
    transforms.Resize(data_res),
    transforms.ToTensor(),
])

dataset = load_dataset(transform=transform)
dataset.resolution = data_res
dataloader = DataLoader(dataset, batch_size=16, num_workers=torch.get_num_threads(), pin_memory=True)

pca_acts = pd.read_csv(f'{ckpt_dir}/pca_acts.csv')
pca_acts['label'] = 'unknown'
pca_acts['iou'] = 0
print(pca_acts.head())

# load unit stats
rq = compute_rq(model, dataset, default_layer, resdir, args)

with open('../json/test_dev.json', 'r') as f:
    test_data = json.load(f)

dft_img = '.' + test_data[0]['patch_dir']
patch_height = '200px'
patch_viewer_layout = {
    'dragmode': "drawclosedpath",
    'margin': dict(l=0, r=0, b=0, t=20, pad=0),
    'newshape': dict(opacity=0.8, line=dict(color="yellow", width=3)),
}
patch_viewer = px.imshow(io.imread(dft_img), binary_string=True)
patch_viewer.update_layout(**patch_viewer_layout)
patch_viewer.update_xaxes(showticklabels=False)
patch_viewer.update_yaxes(showticklabels=False)

plot_height = 300
pca_plot_args = dict(x='x', y='y', color="label",
                     hover_data={'label': True, 'x': False, 'y': False, 'iou': ':.2f'})
pca_plot_layouts = dict(
    legend=dict(
        yanchor='bottom',
        y=0.01,
        xanchor="right",
        x=0.99
    ),
    height=plot_height,
    margin=dict(l=0, r=0, b=0, t=20, pad=0, autoexpand=True),
    clickmode='event+select'
)

pca_plot = px.scatter(pca_acts, **pca_plot_args)
pca_plot.update_layout(**pca_plot_layouts)
pca_plot.update_xaxes(showticklabels=False, title_text="comp-1")
pca_plot.update_yaxes(showticklabels=False, title_text="comp-2")

labels_dropdown = {
    'tissue': 'tissue',
    'calcification': 'calcification',
    'mass': 'mass'
}

patch_viewer = dbc.Card(
    id="patchbox",
    children=[
        dbc.CardHeader(html.H3("Query Units")),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="patch",
                                    figure=patch_viewer,
                                    config={"modeBarButtonsToAdd": ["eraseshape"],
                                            "modeBarButtonsToRemove": ['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d',
                                                                       'autoScale2d', 'resetScale2d'],
                                            'displaylogo': False},
                                    style={'height': patch_height}
                                ),
                            ], width=4
                        ),
                    ]
                )

            ]
        ),
        dbc.CardFooter(
            [
                dcc.Store(
                    id="image_files",
                    data={"files": test_data, "current": 0},
                ),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Previous image", id="previous", outline=True),
                        dbc.Button("Next image", id="next", outline=True),
                    ],
                    size="lg",
                    style={"width": "100%"},
                ),
            ]
        ),
    ], style={}
)


button_height = '35px'
button_width = '80px'
blank_width = '160px'
label_unit_utils = html.Div([
    dbc.Row([
        dbc.Col(
            dcc.Input(id='input', value='', style={'height': button_height, 'width': blank_width})
        ),
        dbc.Col(
            html.Button('add', id='submit', style={'height': button_height, 'width': button_width})
        )
    ], style={"margin-bottom": "15px"}),

    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id="label-dropdown",
                options=[
                    {"label": k, "value": v} for k, v in labels_dropdown.items()
                ],
                value='tissue',
                clearable=False,
                style={'height': button_height, 'width': blank_width}
            )
        ),
        dbc.Col(
            html.Button("update", id="confirm-label", style={'height': button_height, 'width': button_width})
        )
    ]),
], style={'display': 'inline-block'})


unit_vis = dbc.Card(
    children=[
        dbc.CardHeader(html.H3("Label Units")),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="scatter",
                                    figure=pca_plot,
                                    config={'displayModeBar': False},
                                ),
                                dcc.Store(
                                    id="pca_df",
                                    data=pca_acts[['label', 'x', 'y', 'iou']].to_dict(),
                                ),
                                dcc.Store(
                                    id="unit_ious",
                                    data=[0 for _ in range(num_units)],
                                ),
                            ], width=7
                        ),

                        dbc.Col(
                            label_unit_utils,
                            width=5
                        )
                    ]
                )
            ]
        ),
    ], style={}
)

report = dbc.Card(
    children=[
        dbc.CardHeader(html.H3("Report")),
        dbc.CardBody(
            [
                dbc.Col(html.Pre(id='report', style={})) # {"height": '300px', "overflowY": "scroll"}
            ]
        ),
    ], style={}
)

app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(
                [
                    dbc.Row(
                        [
                            dbc.Col(patch_viewer),
                        ],
                    ),

                    dbc.Row(
                        [
                            dbc.Col(unit_vis),
                        ],
                    ),
                ], width=8
            ),
            dbc.Col(report, width=4)
        ])
    ]
)


@app.callback(
    [Output("image_files", "data"), Output("patch", "figure")],
    [
        Input("previous", "n_clicks"),
        Input("next", "n_clicks"),
    ],
    State("image_files", "data"),
)
def browse_image(
        previous_n_clicks,
        next_n_clicks,
        image_files_data,
):
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    image_index_change = 0
    if cbcontext == "previous.n_clicks":
        image_index_change = -1
    if cbcontext == "next.n_clicks":
        image_index_change = 1
    image_files_data["current"] += image_index_change
    image_files_data["current"] %= len(image_files_data["files"])
    if image_index_change != 0:
        filename = '.' + image_files_data["files"][image_files_data["current"]]['patch_dir']
        img = io.imread(filename)
        fig = px.imshow(img, binary_string=True)
        fig.update_layout(
            **patch_viewer_layout
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return image_files_data, fig

    else:
        return dash.no_update


def iou_tensor(candidate: torch.Tensor, example: torch.Tensor):
    intersection = (candidate & example).float().sum((0, 1))
    union = (candidate | example).float().sum((0, 1))

    iou = intersection / (union + 1e-9)
    return iou.item()


@app.callback(
     Output("unit_ious", 'data'),
    [Input("patch", "relayoutData")],
)
def compute_unit_ious(relayout_data):
    if relayout_data is None or 'shapes' not in relayout_data.keys():
        return dash.no_update
    else:
        acts = model.retained_layer(default_layer).cpu()
        ivsmall = imgviz.ImageVisualizer((data_res, data_res), source=dataset, quantiles=rq, level=rq.quantiles(quantile))
        masks = [ivsmall.pytorch_mask(acts, (0, u)) for u in range(num_units)]

        shapes = relayout_data["shapes"]
        image_shape = (data_res, data_res)
        shape_args = [
            {"width": image_shape[1], "height": image_shape[0], "shape": shape}
            for shape in shapes
        ]
        shape_layers = [(n + 1) for n, _ in enumerate(shapes)]
        annot = shape_utils.shapes_to_mask(shape_args, shape_layers)
        contours, _ = cv2.findContours(annot, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        gt_mask = np.zeros(image_shape)
        cv2.fillPoly(gt_mask, pts=[contours[0]], color=(1, 1))

        ious = [iou_tensor(mask, torch.from_numpy(gt_mask) > 0) for mask in masks]
        print('max iou score:', max(ious))
        return ious


def px_fig2array(fname=None):
    if fname is None:
        img = Image.open('data/tmp.png').convert("L")
    else:
        img = Image.open(fname).convert("L")

    img_np = np.asarray(img)

    return img_np


@app.callback(
    [Output("scatter", 'figure'), Output("pca_df", 'data'),  Output('report', 'children')],
    [Input("label-dropdown", "value"), Input('confirm-label', 'n_clicks')],
    [State("unit_ious", 'data'), State("pca_df", 'data')]
)
def update_plot(label, n_click, unit_ious, pca_df):
    label = dash.callback_context.inputs['label-dropdown.value']
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'confirm-label' in changed_id:
        pca_df = pd.DataFrame.from_dict(pca_df)
        print(pca_df.head())
        cur_labels = np.array(pca_df['label'].tolist())
        cur_ious = np.array(pca_df['iou'].tolist())
        new_ious = np.array(unit_ious)

        mask_th = new_ious > iou_th
        mask_up = new_ious > cur_ious
        mask = np.logical_and(mask_th, mask_up)

        updated_labels = cur_labels
        updated_ious = []
        for i, update in enumerate(mask):
            if update:
                updated_labels[i] = label
                updated_ious.append(new_ious[i].item())
            else:
                updated_ious.append(cur_ious[i])

        updated_pca_df = pca_df.copy()
        updated_pca_df['label'] = updated_labels
        updated_pca_df['iou'] = updated_ious

        fig = px.scatter(updated_pca_df, **pca_plot_args)
        fig.update_layout(**pca_plot_layouts)
        fig.update_xaxes(showticklabels=False, title_text="comp-1")
        fig.update_yaxes(showticklabels=False, title_text="comp-2")

        grouped_df = updated_pca_df.groupby('label')
        cur_label_count = grouped_df.size().tolist()
        mean_df = grouped_df.mean().reset_index()

        cur_labels = mean_df['label'].tolist()
        cur_mean_act = mean_df['iou'].tolist()

        grouped_mean_iou = {k: {'num':w, 'iou':round(v, 4)} for k, v, w in zip(cur_labels, cur_mean_act, cur_label_count)}

        return fig, updated_pca_df.to_dict(), json.dumps(grouped_mean_iou, indent=2)
    else:
        return dash.no_update


@app.callback(Output('label-dropdown', 'options'),
              [Input('input', 'value'), Input('submit', 'n_clicks')],
              [State('label-dropdown', 'options')]
              )
def update_label_dropdown(new_value, new_submission, current_options):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit' in changed_id:
        print('submitted')
        if not new_value:
            return current_options

        current_options.append({'label': new_value, 'value': new_value})
        return current_options
    else:
        return dash.no_update


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050)