import plotly
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

def _divmod(dividend,divisor):
    if dividend>=0:
        res = divmod(dividend,divisor)
        return (res[0]+1,res[1])
    dividend=np.abs(dividend)
    res = divmod(dividend,divisor)
    return (-res[0]-1,res[1])

def fold(L:list,rstd=1)->list:
    avg=np.mean(L)
    std=np.std(L)*rstd
    return [ _divmod(x-avg,std) for x in L]

def HorizonGraph(x,y,norm=True):
    if norm:
        y = (np.array(y)-min(y))/(max(y)-min(y))
    height = np.std(y)
    L = fold(y)
    layers,values = zip(*L)
    num_layer = len(np.unique([np.abs(x) for x in list(layers)]))
    alpha = 1.0/num_layer
    x = df.date.tolist()
    x_rev=x[::-1]
    data=[]
    for layer in np.unique(layers):
        if layer>0:
            temp=[t if np.sign(t[0])==np.sign(layer) else (0,0) for t in L]
            y_upper=[height if t[0]!=0 and t[0]>layer else t[1] if t[0]==layer  else 0 for t in temp]    
            y_lower=len(x)*[0]
            #y_lower=y_lower[::-1]
            y_upper=y_upper[::-1]
            data.append(go.Scatter(x=x+x_rev,y=y_lower+y_upper,fill='tozerox',fillcolor='rgba(0,0,150,{})'.format((layer)*alpha),
                                line=dict(color='rgba(255,255,255,0)'),showlegend=False,name="mean + {}*std".format(int(layer)),))
        else:
            temp=[t if np.sign(t[0])==np.sign(layer) else (0,0) for t in L]
            y_upper=len(x)*[height]
            y_lower=[0 if t[0]!=0 and t[0]<layer else height-t[1] if t[0]==layer else height for t in temp]
            y_lower=y_lower[::-1]
            data.append(go.Scatter(x=x+x_rev,y=y_upper+y_lower,fill='tozerox',fillcolor='rgba(150,0,0,{})'.format((-layer)*alpha),
                    line=dict(color='rgba(255,255,255,0)'),showlegend=False,name="mean - {}*std".format(-int(layer)),))
    fig = go.Figure(data=data)
    return fig