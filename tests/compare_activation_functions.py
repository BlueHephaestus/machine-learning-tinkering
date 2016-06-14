from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv1(x):
    return (sigmoid(x))*(1-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_deriv1(x):
    return (1/(np.cosh(x)))**2



N = 500
x = np.linspace(-10, 10, N)
y0 = sigmoid(x)
y1 = sigmoid_deriv1(x)
y2 = tanh(x)
y3 = tanh_deriv1(x)

# Create a trace
trace0 = go.Scatter(
    x = x,
    y = y0,
    name="sigmoid"
)
trace1 = go.Scatter(
    x = x,
    y = y1,
    name="sigmoid prime"
)
trace2 = go.Scatter(
    x = x,
    y = y2,
    name="tanh"
)
trace3 = go.Scatter(
    x = x,
    y = y3,
    name="tanh prime"
)

#data = [trace0, trace1, trace2, trace3]
fig = tools.make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True, subplot_titles=('Sigmoid', 'Sigmoid Prime', 'Tanh', 'Tanh Prime'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 2)

# Plot and embed in ipython notebook!
#py.iplot(data, filename='basic-line')
# Open new tab in browser
plot_url = py.plot(fig, filename='basic-line')
