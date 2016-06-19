import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

f = open('test_output.txt', 'r')

for config in f.readlines():
    config_results = json.loads(config)
    
    
N = 50
x = np.linspace(0, 50, N)
y0 = 
y1 = 

# Create a trace
trace0 = go.Scatter(
    x = x,
    y = y0,
    name="sigmoid"
)

data = [trace0, trace1, trace2, trace3, trace4]
#fig = tools.make_subplots(rows=2

fig = tools.make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True, subplot_titles=('Sigmoid', 'Sigmoid Prime', 'Tanh', 'Tanh Prime'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 2)
# Plot and embed in ipython notebook!
#py.iplot(data, filename='basic-line')
# Open new tab in browser
plot_url = py.plot(data, filename='basic-line')

