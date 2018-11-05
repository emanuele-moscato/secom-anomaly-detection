from flask import Flask, send_from_directory
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, Event
from plotly import graph_objs as go
from app_components import ProcessIndicator, train_model, parse_content, plot_roc_curve
from sklearn.externals import joblib
import os

MODEL_DIR = '../models/'

server = Flask('Dashboard', static_url_path='')

app = dash.Dash(server=server)

@server.route('/static/style.css')
def serve_stylesheet():
    return server.send_static_file('style.css')

app.css.append_css({
    'external_url': '/static/style.css'
})
    
@server.route('/favicon.ico')
def favicon():
    return server.send_static_file('400x400SML-01.png')
    
app.scripts.config.serve_locally = True

app.config['suppress_callback_exceptions']=True

training_indicator = ProcessIndicator()

app.layout = html.Div(children=[
        html.H1("Machine learning for the manufacturing industry"),
        dcc.Upload(
            id='upload-training',
            children=html.A('Upload training data'),
            style={
                'width': '350px',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin-left': 'auto',
                'margin-right': 'auto',
                'margin-top': '10px',
                'margin-bottom': '10px'
            },
            multiple=True
        ),
        html.Div(
            id='training-container',
            children=[
                dcc.Interval(
                    id='input-interval-component',
                    interval=1000, # in milliseconds
                    n_intervals=0    
                ),
                html.Div(
                    id='training-result'
                ),
                html.Div(
                    id='training-done-container'
                )
            ]
        ),
        dcc.Upload(
            id='upload-test',
            children=html.A('Upload test data'),
            style={
                'width': '350px',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin-left': 'auto',
                'margin-right': 'auto',
                'margin-top': '10px',
                'margin-bottom': '10px'
            },
            multiple=True
        ),
        html.Div(
            id='test-container',
            children=[
                dcc.Interval(
                    id='output-interval-component',
                    interval=1000, # in milliseconds
                    n_intervals=0    
                ),
                html.Button(
                    'Reset',
                    id='reset-button'  
                ),
                html.Div(
                    id='test-result',
                    style={
                        'width': '70%',
                        'textAlign': 'center',
                        'margin-left': 'auto',
                        'margin-right': 'auto',
                        'margin-top': '10px',
                        'margin-bottom': '10px'
                    },
                )
            ]
        ),
    ],
    style={'text-align':"center"}
)


@app.callback(Output('training-result', 'children'),
              [Input('upload-training', 'contents')],
              [State('upload-training', 'filename'),
               State('upload-training', 'last_modified')])
def update_training_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents:
        training_indicator.lock()
        training_data = parse_content(list_of_contents[-1])
        train_model(training_data)
        training_indicator.unlock()
    else:
        pass

@app.callback(
    Output('training-done-container', 'children'),
    events=[Event('input-interval-component', 'interval')])
def display_training_status():
    if not training_indicator.is_empty():
        if training_indicator.is_training():
            return html.P("Training...")
        else:
            return html.P("Training done!")

@app.callback(Output('test-result', 'children'),
              [Input('upload-test', 'contents')],
              [State('upload-test', 'filename'),
               State('upload-test', 'last_modified')])
def update_training_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents:
        test_data = parse_content(list_of_contents[-1])
        
        rfc = joblib.load(os.path.join(MODEL_DIR, 'trained_model.joblib'))
        
        roc_plot = plot_roc_curve(rfc, test_data)
        
        return dcc.Graph(figure=roc_plot)
    else:
        pass

@app.callback(
    Output('upload-training', 'contents'),
    events=[Event('reset-button', 'click')])
def display_training_status():
    training_indicator.reset()

if __name__=='__main__':
    app.server.run(debug=True, port=8888, processes=True)