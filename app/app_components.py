from time import sleep
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_curve
import plotly.graph_objs as go
import base64
import io
import pandas as pd
import os

DATA_DIR = '../data/'
MODEL_DIR = '../models/'

class ProcessIndicator(object):
    def __init__(self, filename='process_indicator.txt'):
        self.filename = filename
        with open(self.filename, 'w') as f:
            f.write('no file')

    def lock(self):
        with open(self.filename, 'w') as f:
            f.write('training')

    def unlock(self):
        with open(self.filename, 'w') as f:
            f.write('trained')
    
    def reset(self):
        with open(self.filename, 'w') as f:
            f.write('no file')

    def is_empty(self):
        return open(self.filename, 'r').read() == 'no file'

    def is_training(self):
        return open(self.filename, 'r').read() == 'training'
        
def parse_content(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    data = io.StringIO(decoded.decode('utf-8'))
    
    return data
    

def train_model(training_data):
    sleep(1)
    rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    data_df = pd.read_csv(training_data)
    
    rfc.fit(data_df.drop('label', axis=1), data_df['label'])
    
    joblib.dump(rfc, os.path.join(MODEL_DIR, 'trained_model.joblib'))

def plot_roc_curve(model, test_data_file):
    test_data = pd.read_csv(test_data_file)
    
    fpr, tpr, thresholds = roc_curve(
        test_data['label'],
        model.predict(test_data.drop('label', axis=1))
    )

    trace = go.Scatter(
        x=fpr,
        y=tpr
    )
    
    data = [trace]
    
    layout = go.Layout(
        xaxis = dict(
            title='FPR'
        ),
        yaxis = dict(
            title='TPR'
        ),
        title='ROC curve'
    )
    
    return go.Figure(data=data, layout=layout)