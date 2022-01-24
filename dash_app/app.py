import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
from skimage import io
import json
from PIL import Image
import shape_utils
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

labels_dropdown = {
    'tissue': 'tissue',
    'calcification': 'calcification',
    'mass': 'mass'
}

conv_layers = {'conv1_1': 'features.conv1_1', 'conv1_2': 'features.conv1_2',
               'conv2_1': 'features.conv2_1', 'conv2_2': 'features.conv2_2',
               'conv3_1': 'features.conv3_1', 'conv3_2': 'features.conv3_2', 'conv3_3': 'features.conv3_3',
               'conv4_1': 'features.conv4_1', 'conv4_2': 'features.conv4_2', 'conv4_3': 'features.conv4_3',
               'conv5_1': 'features.conv5_1', 'conv5_2': 'features.conv5_2', 'conv5_3': 'features.conv5_3'}

default_layer = 'features.conv5_3'

# decalare model settings
quantile = 0.95
args = EasyDict(model='vgg16_bn', dataset='breast_cancer', quantile=quantile)

units_dict = {
    'vgg16_bn': {"conv1": 64, "conv2": 128, "conv3": 256, "conv4": 512, "conv5": 512},
}

resdir = 'results/%s-%s-%s-%s' % (args.model, args.dataset, default_layer, int(args.quantile * 1000))

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device=device)
model.retain_layer(default_layer)

# load dataset
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

dataset = load_dataset()
dataloader = DataLoader(dataset, batch_size=128, num_workers=0, pin_memory=True)

# load unit stats
rq = compute_rq(model, dataset, default_layer, resdir, args)

# compute acts for the test set
acts = compute_act(model, dataloader, default_layer, device, resdir)

# compute quantile embedding for the test set
quantile_table = rq.readout()
quantile_mat = compute_act_quantile(quantile_table, acts, resdir)

# tsne plot
data_columns = ['d' + str(i) for i in range(acts.shape[0])]
df = pd.DataFrame(quantile_mat.T, columns=data_columns)

tsne_df = cluster_units(df, data_columns)
scatter = px.scatter(tsne_df,
                     x="tsne-2d-one",
                     y="tsne-2d-two",
                     color="label",
                     hover_data={'unit': True, 'label':True, 'tsne-2d-one': False, 'tsne-2d-two': False,'act': ':.2f'})

scatter.update_layout(
    legend=dict(
        yanchor='bottom',
        y=0.01,
        xanchor="right",
        x=0.99
    )
)
scatter.update_layout(
    height=300,
    margin=dict(l=0, r=0, b=0, t=0, pad=0, autoexpand=False),
)
scatter.update_layout(clickmode='event+select')
scatter.update_xaxes(showticklabels=False, title_text="")
scatter.update_yaxes(showticklabels=False, title_text="")

# bar chart
bar_chart_data = [{'unit': i, 'value': (512 - i) * 0.01} for i in range(512)]
df_bar = pd.DataFrame(bar_chart_data)
bar_fig = px.bar(bar_chart_data, x='unit', y='value')
bar_fig.show()
bar_fig.update_layout(
    height=50,
    margin=dict(l=0, r=0, b=0, t=0, pad=0),
)
bar_fig.update_xaxes(showticklabels=False, showgrid=False, title_text="")
bar_fig.update_yaxes(showticklabels=False, showgrid=False, title_text="")

external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/image_annotation_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

with open('../json/whole_test_img.json', 'r') as f:
    filelist = json.load(f)

server = app.server

fig = px.imshow(io.imread(filelist[0]), binary_string=True)
fig.update_traces(hoverinfo='none', hovertemplate=None)
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=0, pad=0),
)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

placeholder = px.imshow(io.imread('./assets/placeholder.png'), binary_string=True)
placeholder.update_layout(
    dragmode="drawclosedpath",
    margin=dict(l=0, r=0, b=0, t=0, pad=0),
)
placeholder.update_xaxes(showticklabels=False)
placeholder.update_yaxes(showticklabels=False)

image_annotation_card = dbc.Card(
    id="imagebox",
    children=[
        dbc.CardHeader(html.H3("Full Mammogram")),
        dbc.CardBody(
            [
                dcc.Graph(
                    id="graph",
                    figure=fig,
                    style={'height': '575px'}
                )
            ]
        ),
        dbc.CardFooter(
            [
                dcc.Markdown(
                    "Click to select a ROI to feed in our patch-based model"
                ),
                dcc.Store(
                    id="image_files",
                    data={"files": filelist, "current": 0},
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
    ],
)


patch_height = '140px'
selected_patch = dbc.Card(
    id="patchbox",
    children=[
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="patch",
                                    figure=placeholder,
                                    config={"modeBarButtonsToAdd": ["eraseshape"],
                                            "modeBarButtonsToRemove": ['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d',
                                                                       'autoScale2d', 'resetScale2d'],
                                            'displaylogo': False},
                                    style={'height': patch_height}
                                ),

                                dcc.Graph(
                                    id="unit-act",
                                    figure=placeholder,
                                    config={'displayModeBar': False,'displaylogo': False},
                                    style={'height': patch_height}
                                ),
                            ],
                        ),
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="act",
                                    figure=placeholder,
                                    config={"modeBarButtonsToRemove": ['pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d',
                                                                       'autoScale2d', 'resetScale2d'],
                                            'displaylogo': False},
                                    style={'height': patch_height}
                                ),

                                dcc.Graph(
                                    id="max-unit-act",
                                    figure=placeholder,
                                    config={'displayModeBar': False,'displaylogo': False},
                                    style={'height': patch_height}
                                ),
                            ]
                        ),
                        dbc.Col(html.Pre(id='act-stats', style={"height": '280px', "overflowY": "scroll"}))
                    ]
                )

            ]
        ),
    ],
)

scatter_plot = dbc.Card(
    [
        dbc.CardBody(
            [
                dbc.Row(
                    [

                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="scatter",
                                    figure=scatter,
                                    config={'displayModeBar': False},
                                ),
                                dcc.Store(
                                    id="scatter-data",
                                    data=tsne_df[['label', 'act', 'unit', 'tsne-2d-one', 'tsne-2d-two']].copy().to_dict(),
                                ),
                            ], width=8
                        ),

                        dbc.Col(
                            html.Pre(id='after-act-stats', style={"height": '300px', "overflowY": "scroll"}), width=4
                        ),
                    ]
                )
            ]
        ),
        dbc.CardFooter(
            [
                html.Div(
                    [
                        # We use this pattern because we want to be able to download the
                        # annotations by clicking on a button
                        html.A(
                            id="download",
                            download="annotations.json",
                            # make invisible, we just want it to click on it
                            style={"display": "none"},
                        ),
                        dbc.Button(
                            "Download annotations", id="download-button", outline=True,
                        ),
                        html.Div(id="dummy", style={"display": "none"}),
                        dbc.Tooltip(
                            "You can download the annotated data in a .json format by clicking this button",
                            target="download-button",
                        ),
                    ],
                )
            ]
        ),
    ], style={"margin-top": "10px"}
)

button_height = '35px'
div_for_plot = html.Div([
    html.Div([
        dcc.Input(id='input', value='', style={'height': button_height}),
        html.Button('Add label', id='submit', style={'height': button_height,
                                                     "margin-right": "15px"})
    ]),

    html.Div([
        dcc.Dropdown(
            id="label-dropdown",
            options=[
                {"label": k, "value": v} for k, v in labels_dropdown.items()
            ],
            value='tissue',
            clearable=False,
            style={'height': button_height, 'width': '15vw'}
        ),
    ]),

    html.Div(
        html.Button("Label units", id="confirm-label", style={'height': button_height}),
    ),

    html.Div(
        dcc.Loading(id="loading", type="default", children=[html.Div([html.Div(id="loading-2")])],
                    style={'height': button_height, "margin-left": "100px"})
    )

], style=dict(display='flex'))


app.layout = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(image_annotation_card),
                        dbc.Col(
                            children=[
                                # dcc.Dropdown(
                                #     id="layer-dropdown",
                                #     options=[
                                #         {"label": k, "value": v} for k, v in conv_layers.items()
                                #     ],
                                #     value=default_layer,
                                #     clearable=False,
                                # ),
                                html.Div(
                                    dcc.Graph(
                                        id='bar',
                                        figure=bar_fig,
                                        config={'displayModeBar': False},
                                    ),
                                    style={"width": '675px', "overflowX": "scroll"}
                                ),
                                selected_patch,
                                div_for_plot,
                                scatter_plot,
                            ],
                        )
                    ],
                ),
            ],
            fluid=True,
        ),
    ]
)


@app.callback(Output('label-dropdown', 'options'), 
              [Input('input', 'value'), Input('submit', 'n_clicks')], 
              [State('label-dropdown', 'options')]
)
def update_label_dropdown(new_value, new_submission, current_options):
    print('here')
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print([p['prop_id'] for p in dash.callback_context.triggered])
    if 'submit' in changed_id:
        print('submitted')
        if not new_value:
            return current_options

        current_options.append({'label': new_value, 'value': new_value})
        return current_options
    else:
        return dash.no_update
    

@app.callback(
    Output("bar", 'figure'),
    [Input("patch", "relayoutData")], 
)
def update_bar(relayout_data):
    img = px_fig2array()
    img_input = transform(Image.fromarray(img))
    model(img_input.unsqueeze(1).to(device))
    acts = model.retained_layer(default_layer).cpu()
    acts = acts.view(acts.shape[0], acts.shape[1], -1).mean(2) # or .max(2)[0]
    acts_quantile = compute_act_quantile(quantile_table, acts, resdir)
    
    units_num_sorted = [i[0] for i in sorted(enumerate(acts_quantile.tolist()[0]), reverse = True, key=lambda x:x[1])]
    units_acts_sorted = [i[1] for i in sorted(enumerate(acts_quantile.tolist()[0]), reverse = True, key=lambda x:x[1])]
    
    act_bar_chart_data = [{'unit': unit, 'value': act_value, 'rule': 'act'} 
                          for unit, act_value in zip(units_num_sorted, units_acts_sorted)]
    
    if relayout_data is None or 'shapes' not in relayout_data.keys():
        bar_chart_data = act_bar_chart_data
    else:
        num_units = 512
        acts = model.retained_layer(default_layer).cpu()
        ivsmall = imgviz.ImageVisualizer((128, 128), source=dataset, quantiles=rq, level=rq.quantiles(quantile))
        masks = [ivsmall.pytorch_mask(acts, (0, u)) for u in range(num_units)]
        
        shapes = relayout_data["shapes"]
        image_shape = (128, 128)
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
        reindexed_ious = [ious[i] for i in units_num_sorted]
        iou_bar_chart_data = [{'unit': unit, 'value': iou_value, 'rule': 'iou'} 
                              for unit, iou_value in zip(units_num_sorted, reindexed_ious)]
        
        bar_chart_data = act_bar_chart_data + iou_bar_chart_data
    
    bar_fig = px.bar(bar_chart_data, x='unit', y='value', color='rule', barmode='group')

    bar_fig.update_layout(
        xaxis_type = 'category',
        showlegend=False,
        height=95,
        width=20000,
        margin=dict(l=0, r=0, b=10, t=0, pad=0, autoexpand=False), 
    )
    bar_fig.update_xaxes(showticklabels=False, showgrid=False, title_text="", rangeslider_visible=False)
    bar_fig.update_yaxes(showticklabels=False, showgrid=False, title_text="")
    
    return bar_fig


@app.callback(
    [Output('unit-act', 'figure'), Output('max-unit-act', 'figure')],
    [Input('scatter', 'clickData')]
)
def show_unit_act(click_data):
    if click_data is None:
        return dash.no_update

    selected_unit = click_data['points'][0]['customdata'][0]

    img = px_fig2array()
    img_input = transform(Image.fromarray(img))
    model(img_input.unsqueeze(1).to(device))
    acts = model.retained_layer(default_layer).cpu()

    ivsmall = imgviz.ImageVisualizer((128, 128), source=dataset, quantiles=rq, level=rq.quantiles(quantile))
    patch_np = ivsmall.masked_image(img_input, acts, (0, selected_unit))
    
    fig = px.imshow(patch_np, binary_string=True)
    fig.update_layout(
        title=f'unit {selected_unit}', title_x=0.5,
        margin=dict(l=0, r=0, b=10, t=20, pad=0),
        font=dict(
            size=8,
        )
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    max_img = px_fig2array(f'data/unit{selected_unit}_max.png')
    max_input = transform(Image.fromarray(max_img))
    model(max_input.unsqueeze(1).to(device))
    acts = model.retained_layer(default_layer).cpu()

    max_np = ivsmall.masked_image(max_input, acts, (0, selected_unit))
    max_fig = px.imshow(max_np, binary_string=True)
    max_fig.update_layout(
        title=f'unit {selected_unit} max', title_x=0.5,
        margin=dict(l=0, r=0, b=10, t=20, pad=0),
        font=dict(
            size=8,
        )
    )
    max_fig.update_xaxes(showticklabels=False)
    max_fig.update_yaxes(showticklabels=False)
    
    return fig, max_fig


@app.callback(
    [Output("scatter", 'figure'), Output("scatter-data", 'data'),
     Output("loading", "children"), Output('after-act-stats', 'children')],
    [Input("label-dropdown", "value"), Input('confirm-label', 'n_clicks')], 
    [State("scatter-data", 'data')]
)

def update_plot(label, n_click, scatter_data):
    label = dash.callback_context.inputs['label-dropdown.value']
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'confirm-label' in changed_id:
        img = px_fig2array()

        img_input = transform(Image.fromarray(img))
        model(img_input.unsqueeze(1).to(device))
        acts = model.retained_layer(default_layer).cpu()
        acts = acts.view(acts.shape[0], acts.shape[1], -1)
        acts = acts.mean(2) 
        
        acts_quantile = compute_act_quantile(quantile_table, acts, resdir)
        acts_quantile = acts_quantile.tolist()[0]
        
        dataframe = pd.DataFrame.from_dict(scatter_data).copy()
        cur_acts = dataframe['act'].tolist()
        cur_labels = dataframe['label'].tolist()
        
        updated_acts = []
        updated_labels = []
        updated = []
        
        for cur_label, cur_act, new_act in zip(cur_labels, cur_acts, acts_quantile):
            if label == cur_label:
                updated_acts.append(max(cur_act, new_act))
                updated_labels.append(cur_label)
            else:
                if new_act > cur_act:
                    updated_acts.append(new_act)
                    updated_labels.append(label)
                else:
                    updated_acts.append(cur_act)
                    updated_labels.append(cur_label)
            
            updated.append(new_act > cur_act)
        
        for idx, update in enumerate(updated):
            if update:
                im = Image.fromarray(img)
                im.save(f"data/unit{idx}_max.png")
        
        dataframe['label'] = updated_labels
        dataframe['act'] = updated_acts
          
        fig = px.scatter(dataframe, 
                         x="tsne-2d-one", 
                         y="tsne-2d-two", 
                         color="label",
                         hover_data={'unit': True, 'label':True, 'tsne-2d-one': False, 'tsne-2d-two': False,'act': ':.2f'})
       
        fig.update_layout(
            legend=dict(
                yanchor='bottom',
                y=0.01,
                xanchor="right",
                x=0.99
            )
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, b=0, t=0, pad=0, autoexpand=False),
            clickmode='event+select'
        )
        fig.update_xaxes(showticklabels=False, title_text="")
        fig.update_yaxes(showticklabels=False, title_text="")

        df = dataframe.copy()
        df['label'] = updated_labels
        df['act'] = acts_quantile
        grouped_df = df.groupby('label')
        cur_label_count = grouped_df.size().tolist()
        mean_df = grouped_df.mean()
        mean_df = mean_df.reset_index()
        
        cur_labels = mean_df['label'].tolist()
        cur_mean_act = mean_df['act'].tolist()
        
        cur_act_avg_by_label = {k: {'num':w, 'mean':round(v, 4)} for k, v, w in zip(cur_labels, cur_mean_act, cur_label_count)}
        
        return fig, dataframe.to_dict(), '', json.dumps(cur_act_avg_by_label, indent=2)
    else:
        return dash.no_update

    
@app.callback(
    [Output('patch', 'figure'), Output('act-stats', 'children')],
    Input('graph', 'clickData'),
    [State("image_files", "data"), State("scatter-data", 'data')]
)
def display_click_data(clickData, image_files_data, scatter_data):
    if clickData is None:
        title = ''
        fig = px.imshow(io.imread('./assets/placeholder.png'), binary_string=True)
        fig.update_layout(
            title=title, title_x=0.5,
            dragmode="drawclosedpath",
            margin=dict(l=0, r=0, b=0, t=20, pad=0),
            newshape=dict(opacity=0.8, line=dict(color="yellow", width=4)),
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig, dash.no_update
    else:
        filename = image_files_data["files"][image_files_data["current"]]
        img = io.imread(filename)
        img_size = 128 // 2

        y_min, y_max = img_size, img.shape[0] - img_size
        x_min, x_max = img_size, img.shape[1] - img_size

        cx = clickData['points'][0]['x']
        cy = clickData['points'][0]['y']

        cx = max(cx, x_min)
        cx = min(cx, x_max)

        cy = max(cy, y_min)
        cy = min(cy, y_max)

        x_start = cx - img_size
        x_end = cx + img_size
        y_start = cy - img_size
        y_end = cy + img_size

        if len(img.shape) == 3:
            patch_np = img[y_start:y_end, x_start:x_end, :]
        elif len(img.shape) == 2:
            patch_np = img[y_start:y_end, x_start:x_end]
        else:
            print(f'unknown image shape {img.shape}')
            
        im = Image.fromarray(patch_np)
        im.save("data/tmp.png")

        fig = px.imshow(patch_np, binary_string=True, binary_compression_level=0)
        input_patch = transform(Image.fromarray(patch_np))

        output = model(input_patch.unsqueeze(1).to(device))
        prob = F.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1)
        
        # get acts quantile
        acts = model.retained_layer(default_layer).cpu()
        acts = acts.view(acts.shape[0], acts.shape[1], -1)
        acts = acts.mean(2)

        acts_quantile = compute_act_quantile(quantile_table, acts, resdir)
        acts_quantile = acts_quantile.tolist()[0]
        
        df = pd.DataFrame.from_dict(scatter_data).copy()
        df['act'] = acts_quantile
        
        grouped_df = df.groupby('label')
        cur_label_count = grouped_df.size().tolist()
        mean_df = grouped_df.mean()
        mean_df = mean_df.reset_index()
        
        cur_labels = mean_df['label'].tolist()
        cur_mean_act = mean_df['act'].tolist()
        
        cur_act_avg_by_label = {k: {'num':w, 'mean':round(v, 4)} for k, v, w in zip(cur_labels, cur_mean_act, cur_label_count)}

        labels = ['normal', 'lesion']
        title = f'{labels[pred.item()]}: {round(torch.max(prob[0]).item(), 3)}'

        fig.update_layout(
            title=title, title_x=0.5,
            dragmode="drawclosedpath",
            margin=dict(l=0, r=0, b=10, t=20, pad=0),
            newshape=dict(opacity=0.8, line=dict(color="yellow", width=3)),
            font=dict(size=8)
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

    return fig, json.dumps(cur_act_avg_by_label, indent=2)
    
    
def px_fig2array(fname=None):
    if fname is None:
        img = Image.open('data/tmp.png').convert("L")
    else:
        img = Image.open(fname).convert("L")

    img_np = np.asarray(img)

    return img_np
    

def iou_tensor(candidate: torch.Tensor, example: torch.Tensor):
    intersection = (candidate & example).float().sum((0, 1))
    union = (candidate | example).float().sum((0, 1))
    
    iou = intersection / (union + 1e-9)
    return iou.item()


@app.callback(
    Output('act', 'figure'),
    [Input("bar", "clickData")]
)
def display_act(clickData):
    ctx = dash.callback_context
        
    if clickData is None:
        return dash.no_update
    else:
        selected_unit = int(clickData['points'][0]['x'])
        img = px_fig2array()
        img_input = transform(Image.fromarray(img))
        model(img_input.unsqueeze(1).to(device))
        
        acts = model.retained_layer(default_layer).cpu()
        ivsmall = imgviz.ImageVisualizer((128, 128), source=dataset, quantiles=rq, level=rq.quantiles(quantile))
        selected_mask = ivsmall.masked_image(img_input, acts, (0, selected_unit))
        fig = px.imshow(selected_mask)
        fig.update_layout(
            title=f'unit {selected_unit}', title_x=0.5,
            margin=dict(l=0, r=0, b=10, t=20, pad=0),
            font=dict(
                size=8,
            )
        )

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        return fig
    

@app.callback(
    [Output("image_files", "data"), Output("graph", "figure")],
    [
        Input("previous", "n_clicks"),
        Input("next", "n_clicks"),
        Input('graph', 'clickData'),
    ],
    State("image_files", "data"),
)
def update_full_image(
        previous_n_clicks,
        next_n_clicks,
        click_data,
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
        filename = image_files_data["files"][image_files_data["current"]]
        img = io.imread(filename)
        fig = px.imshow(img, binary_string=True)
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
        )
        fig.update_traces(hoverinfo='none', hovertemplate=None)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return image_files_data, fig
    
    if cbcontext == 'graph.clickData':
        filename = image_files_data["files"][image_files_data["current"]]
        img = io.imread(filename)
        img_size = 128 // 2

        y_min, y_max = img_size, img.shape[0] - img_size
        x_min, x_max = img_size, img.shape[1] - img_size

        cx = click_data['points'][0]['x']
        cy = click_data['points'][0]['y']

        cx = max(cx, x_min)
        cx = min(cx, x_max)

        cy = max(cy, y_min)
        cy = min(cy, y_max)

        x_start = cx - img_size
        x_end = cx + img_size
        y_start = cy - img_size
        y_end = cy + img_size
        
        fig = px.imshow(img, binary_string=True)
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            shapes=[
                dict(type="rect", xref="x", yref='y',
                     x0=x_start, y0=y_start, x1=x_end, y1=y_end, line_color="yellow", line_width=1),
            ]
        )
        fig.update_traces(hoverinfo='none', hovertemplate=None)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return image_files_data, fig
    
    else:
        return dash.no_update

    
# set the download url to the contents of the annotations-store (so they can bedownloaded from the browser's memory)
app.clientside_callback(
    """
    function(the_store_data) {
        let s = JSON.stringify(the_store_data);
        let b = new Blob([s],{type: 'text/plain'});
        let url = URL.createObjectURL(b);
        return url;
    }
    """,
    Output("download", "href"),
    [Input("scatter-data", "data")],
)


# click on download link via button
app.clientside_callback(
    """
    function(download_button_n_clicks)
    {
        let download_a=document.getElementById("download");
        download_a.click();
        return '';
    }
    """,
    Output("dummy", "children"),
    [Input("download-button", "n_clicks")],
)

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050)
