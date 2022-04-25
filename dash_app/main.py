import math
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
import base64
import numpy as np
import pandas as pd
import cv2
from netdissect import imgviz
from compute_unit_stats import (
    load_model, load_dataset, compute_rq, compute_topk, load_topk_imgs
)
from utils import (
    inference, whole_image_inference, pad_img_row
)
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from netdissect.easydict import EasyDict
import plotly.io as pio

pio.templates.default = "simple_white"

external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/image_annotation_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

data_v = 'version_0'
exp = 'adam_20220318012236'
ckpt_dir = f'../ckpt/{data_v}/vgg16_bn_{exp}'
data_res = 512
num_units = 512
default_layer = 'features.conv5_3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(exp=os.path.basename(ckpt_dir), device=device)
model.retain_layer(default_layer)
labels = ['normal', 'lesion']

# decalare model settings
quantile = 0.99
args = EasyDict(model='vgg16_bn', exp=exp, quantile=quantile)
resdir = f'results/{data_v}/%s-%s-%s-%s' % (args.model, args.exp, default_layer, int(args.quantile * 100))

# iou threshold
iou_th = 0.2

# load dataset
transform = transforms.Compose([
    transforms.Resize(data_res),
    transforms.ToTensor(),
])

dataset = load_dataset(data_v=data_v, transform=transform)
dataset.resolution = data_res
dataloader = DataLoader(dataset, batch_size=16, num_workers=torch.get_num_threads(), pin_memory=True)

pca_acts = pd.read_csv(f'{ckpt_dir}/pca_acts.csv')
pca_acts['label'] = 'unknown'
pca_acts['iou'] = 0
pca_acts['size'] = [5] * num_units
pca_acts['unit'] = range(num_units)

# load unit stats
rq = compute_rq(model, dataset, default_layer, resdir, args)

# load topk
topk = compute_topk(model, dataset, default_layer, resdir)

# load topk imagees
unit_images = load_topk_imgs(model, dataset, rq, topk, default_layer, quantile, resdir)

with open(f'../json/{data_v}/test_dev.json', 'r') as f:
    test_data = json.load(f)

dft_mamm = '.' + test_data[0]['imaging_dir']
cropped_img, lesion_fs = whole_image_inference(dft_mamm, data_res, model, transform)
img_height = '450px'
img_viewer_layout = {
    'margin': dict(l=0, r=0, b=20, t=0, pad=0),
}
img_viewer = px.imshow(cropped_img, binary_string=True)
img_viewer.update_layout(**img_viewer_layout)
img_viewer.update_xaxes(visible=False)
img_viewer.update_yaxes(visible=False)

preview_height = '120px'
preview_layout = {
    'margin': dict(l=0, r=0, b=0, t=0, pad=0),
}
img_row = pad_img_row(lesion_fs, data_res)
preview = px.imshow(img_row, binary_string=True)
preview.update_layout(**preview_layout)
preview.update_xaxes(visible=False)
preview.update_yaxes(visible=False)

place_holder = np.ones((data_res, data_res)) * 255
patch_height = '200px'
patch_viewer_layout = {
    'title': f'prediction:', 'title_x': 0.5,
    'dragmode': "drawclosedpath",
    'margin': dict(l=0, r=0, b=0, t=20, pad=0),
    'newshape': dict(opacity=0.8, line=dict(color="yellow", width=3)),
    'font': dict(size=8)
}
patch_viewer = px.imshow(place_holder, binary_string=True)
patch_viewer.update_layout(**patch_viewer_layout)
patch_viewer.update_xaxes(visible=False)
patch_viewer.update_yaxes(visible=False)

max_act_height = patch_height
max_act_viewer_layout = {
    'title': f'max unit activation:', 'title_x': 0.5,
    'margin': dict(l=0, r=0, b=0, t=20, pad=0),
    'font': dict(size=8)
}
max_act_viewer = px.imshow(place_holder, binary_string=True)
max_act_viewer.update_layout(**max_act_viewer_layout)
max_act_viewer.update_xaxes(visible=False)
max_act_viewer.update_yaxes(visible=False)

mask_height = patch_height
mask_viewer_layout = {
    'title': f'unit with max iou: ', 'title_x': 0.5,
    'margin': dict(l=0, r=0, b=0, t=20, pad=0),
    'newshape': dict(opacity=0.8, line=dict(color="yellow", width=3)),
    'font': dict(size=8)
}
mask_viewer = px.imshow(place_holder, binary_string=True)
mask_viewer.update_layout(**mask_viewer_layout)
mask_viewer.update_xaxes(visible=False)
mask_viewer.update_yaxes(visible=False)

plot_height = 300
pca_plot_args = dict(x='x', y='y', color="label", opacity=0.5, size='size', size_max=5,
                     hover_data={'unit': True, 'label': True, 'x': False, 'y': False, 'iou': ':.2f'})
pca_plot_layouts = dict(
    legend=dict(
        yanchor='top',
        xanchor="right",
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

image_viewer = dbc.Card(
    id="imgbox",
    children=[
        dbc.CardHeader(html.H3("Mammogram")),
        dbc.CardBody(
            [
                dbc.Row(
                    dbc.Col(
                        [
                            dcc.Graph(
                                id='mamm',
                                figure=img_viewer,
                                config={"modeBarButtonsToRemove": ['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d',
                                                                   'autoScale2d', 'resetScale2d'],
                                        'displaylogo': False},
                                style={'height': img_height}
                            )
                        ]
                    )
                ),
                html.Hr(),
                dbc.Row(html.P(id='result', children=[f'{len(lesion_fs)} lesions detected.'])),

                dbc.Row([
                    html.Div([
                        dbc.Col([
                            dcc.Graph(
                                id='preview',
                                figure=preview,
                                config={"modeBarButtonsToRemove": ['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d',
                                                                   'autoScale2d', 'resetScale2d'],
                                        'displaylogo': False},
                                style={'height': preview_height}
                            )
                        ])
                    ], style={'height': preview_height, "width": '400px', "overflowX": "scroll"}
                    )
                ])
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
                        dbc.Button("Previous image", id="previous_m", outline=True),
                        dbc.Button("Next image", id="next_m", outline=True),
                    ],
                    size="lg",
                    style={"width": "100%"},
                ),
            ]
        ),
    ], style={}
)


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

                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="maxact",
                                    figure=max_act_viewer,
                                    config={"modeBarButtonsToAdd": ["eraseshape"],
                                            "modeBarButtonsToRemove": ['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d',
                                                                       'autoScale2d', 'resetScale2d'],
                                            'displaylogo': False},
                                    style={'height': max_act_height}
                                ),
                            ], width=4
                        ),

                        dbc.Col(
                            [
                                dcc.Graph(
                                    id='mask',
                                    figure=mask_viewer,
                                    config={"modeBarButtonsToRemove": ['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d',
                                                                       'autoScale2d', 'resetScale2d'],
                                            'displaylogo': False},
                                    style={'height': mask_height}

                                )
                            ]
                        )
                    ]
                )

            ]
        ),
    ], style={}
)


button_height = '35px'
button_width = '80px'
blank_width = '120px'
label_unit_utils = html.Div([
    dbc.Row([
        dbc.Col(
            dcc.Input(id='input', value='', style={'height': button_height, 'width': blank_width})
        ),

        dbc.Col(
            html.Button('add', id='submit', style={'height': button_height, 'width': button_width})
        ),
    ], style={"margin-bottom": "15px", "margin-top": "10px"}),

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

    dbc.Row([
        dbc.Col(
            html.Div(
                id='topk',
                children=[],
                style={'height': '150px',
                       "width": '240px',
                       "margin-top": "45px",
                       "overflowX": "scroll"}
            ), width=4
        )
    ])
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
                                    data=pca_acts[['label', 'x', 'y', 'iou', 'unit', 'size']].to_dict(),
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
                    image_viewer
                ], width=4
            ),
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
                ], width=6
            ),
            dbc.Col(report, width=2)
        ])
    ],  fluid=True
)


@app.callback(
    [Output("image_files", "data"),
     Output("mamm", "figure"), Output('preview', 'figure'), Output('result', 'children'),
     Output('patch', 'figure')],
    [
        Input("previous_m", "n_clicks"),
        Input("next_m", "n_clicks"),
        Input('preview', 'clickData')
    ],
    State("image_files", "data"),
)
def browse_image(
        previous_n_clicks,
        next_n_clicks,
        click_data,
        image_files_data,
):
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    image_index_change = 0
    if cbcontext == "previous_m.n_clicks":
        image_index_change = -1
    if cbcontext == "next_m.n_clicks":
        image_index_change = 1
    image_files_data["current"] += image_index_change
    image_files_data["current"] %= len(image_files_data["files"])
    if image_index_change != 0:
        filename = '.' + image_files_data["files"][image_files_data["current"]]['imaging_dir']
        cropped_img, lesion_fs = whole_image_inference(filename, data_res, model, transform)

        img_row = pad_img_row(lesion_fs, data_res)

        preview = px.imshow(img_row, binary_string=True)
        preview.update_layout(**preview_layout)
        preview.update_xaxes(visible=False)
        preview.update_yaxes(visible=False)

        fig = px.imshow(cropped_img, binary_string=True)
        fig.update_layout(
            **img_viewer_layout
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return image_files_data, fig, preview, f'{len(lesion_fs)} lesions detected.', dash.no_update

    if cbcontext == "preview.clickData":
        filename = '.' + image_files_data["files"][image_files_data["current"]]['imaging_dir']
        cropped_img, lesion_fs = whole_image_inference(filename, data_res, model, transform)

        x = click_data['points'][0]['x']
        lesion_f_id = x // (data_res + 32)

        patch_f = lesion_fs[lesion_f_id]
        _, y_start, x_start = os.path.basename(patch_f).split('.png')[0].split('_')
        x_start, y_start = int(x_start) * data_res, int(y_start) * data_res
        x_end, y_end = x_start + data_res, y_start + data_res

        fig = px.imshow(cropped_img, binary_string=True)
        fig.update_layout(
            shapes=[
                dict(type="rect", xref="x", yref='y',
                     x0=x_start, y0=y_start, x1=x_end, y1=y_end, line_color="yellow", line_width=1),
            ]
        )
        fig.update_layout(
            **img_viewer_layout
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        patch = io.imread(patch_f)
        im = Image.fromarray(patch)
        im.save('./data/cur_sel.png')
        pred, prob = inference(model, patch, transform, device)
        patch_viewer = px.imshow(patch, binary_string=True)
        labels = ['normal', 'lesion']
        title = f'{labels[pred]}: {round(prob, 3)}'
        patch_viewer_layout['title'] = title
        patch_viewer.update_layout(**patch_viewer_layout)
        patch_viewer.update_xaxes(visible=False)
        patch_viewer.update_yaxes(visible=False)
        return dash.no_update, fig, dash.no_update, dash.no_update, patch_viewer

    return dash.no_update


def iou_tensor(candidate: torch.Tensor, example: torch.Tensor):
    intersection = (candidate & example).float().sum((0, 1))
    union = (candidate | example).float().sum((0, 1))

    iou = intersection / (union + 1e-9)
    return iou.item()


@app.callback(
    [Output("unit_ious", 'data'), Output("mask", "figure"), Output('maxact', 'figure')],
    [Input("patch", "relayoutData"), Input("patch", "figure")],
)
def compute_unit_ious(relayout_data, patch_figure):
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    fname = './data/cur_sel.png'
    img = transform(Image.open(fname))
    ivsmall = imgviz.ImageVisualizer((data_res, data_res), source=dataset, quantiles=rq, level=rq.quantiles(quantile))

    if cbcontext == 'patch.relayoutData':
        if relayout_data is None or 'shapes' not in relayout_data.keys():
            return dash.no_update
        acts = model.retained_layer(default_layer).cpu()
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

        max_unit = np.argmax(np.array(ious))

        max_np = ivsmall.masked_image(img, acts, (0, max_unit))
        max_fig = px.imshow(max_np, binary_string=True)
        max_fig.update_layout(
            title=f'unit {max_unit} max iou: {round(max(ious), 2)}', title_x=0.5,
            margin=dict(l=0, r=0, b=0, t=20, pad=0),
            font=dict(
                size=8,
            )
        )
        max_fig.update_xaxes(visible=False)
        max_fig.update_yaxes(visible=False)

        print('max iou score:', max(ious))

        return ious, max_fig, dash.no_update

    elif cbcontext == 'patch.figure':
        ivsmall = imgviz.ImageVisualizer((data_res, data_res), source=dataset)

        model(img.unsqueeze(1).to(device))
        acts = model.retained_layer(default_layer)
        max_acts = acts.view(512, 32*32).max(1)[0].cpu()
        mask = max_acts > rq.quantiles(quantile)
        non_zero_ids = torch.nonzero(mask)
        max_unit = non_zero_ids[np.argmax(max_acts[mask].numpy())].item()

        max_np = ivsmall.masked_image(img, acts, (0, max_unit), percent_level=0.99)
        max_fig = px.imshow(max_np, binary_string=True)
        max_fig.update_layout(
            title=f'max unit activation', title_x=0.5,
            margin=dict(l=0, r=0, b=0, t=20, pad=0),
            font=dict(
                size=8,
            )
        )
        max_fig.update_xaxes(visible=False)
        max_fig.update_yaxes(visible=False)

        return dash.no_update, dash.no_update, max_fig

    return dash.no_update


def px_fig2array(fname=None):
    if fname is None:
        img = Image.open('data/tmp.png').convert("L")
    else:
        img = Image.open(fname).convert("L")

    img_np = np.asarray(img)

    return img_np

@app.callback(
    Output("topk", "children"),
    [Input('scatter', 'clickData')]
)
def show_topk(click_data):
    if click_data is None:
        return dash.no_update

    unit = click_data['points'][0]['customdata'][0]
    print(unit, unit_images[unit].size)

    topk_imgs = unit_images[unit]
    tmp_name = './data/topk_tmp.png'
    topk_imgs.save(tmp_name)

    topk_base64 = base64.b64encode(open(tmp_name, 'rb').read()).decode('ascii')
    return html.Img(src='data:image/png;base64,{}'.format(topk_base64), style={'height':'85%'})

@app.callback(
    [Output("scatter", 'figure'), Output("pca_df", 'data'),  Output('report', 'children')],
    [Input("label-dropdown", "value"), Input('confirm-label', 'n_clicks'), Input("unit_ious", 'data')],
    [State("pca_df", 'data')]
)
def update_plot(label, n_click, unit_ious, pca_df):
    label = dash.callback_context.inputs['label-dropdown.value']
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'confirm-label' in changed_id:
        pca_df = pd.DataFrame.from_dict(pca_df)
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
        updated_pca_df['unit'] = range(num_units)
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

    if 'unit_ious' in changed_id:
        pca_df = pd.DataFrame.from_dict(pca_df)
        preview_args = pca_plot_args.copy()

        pca_df['size'] = [20 if i > iou_th else 5 for i in unit_ious]
        pca_df['iou'] = unit_ious
        preview_args['size_max'] = math.sqrt(max(pca_df['size']) / min(pca_df['size'])) * 5

        fig = px.scatter(pca_df, **preview_args)
        fig.update_layout(**pca_plot_layouts)
        fig.update_xaxes(showticklabels=False, title_text="comp-1")
        fig.update_yaxes(showticklabels=False, title_text="comp-2")

        return fig, dash.no_update, []

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