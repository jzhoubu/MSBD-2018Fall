import plotly
from matplotlib.pyplot import *
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=False)

def HorizonPlot(df,x,norm=True):
    assert x in df
    fig = tools.make_subplots(rows=df.shape[1]-1, cols=1,shared_xaxes=True)
    xx = df[x].tolist()
    for ind,col in enumerate(df.drop([x],axis=1).columns.tolist()):
        subfig = HorizonGraph(xx,df[col].tolist())
        for i in range(len(subfig.data)):
            fig.append_trace(subfig.data[i], ind+1, 1)
        #fig['layout']['xaxis'+str(ind+1)].update(showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False)
        fig['layout']['yaxis'+str(ind+1)].update(showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False)
    return fig

if __name__=="__main__":
    data=pd.read_csv("../crypto-markets.csv")
    coins_dict={}
    for i in ["BTC","ETH","LTC","NEO","XRP"]:
        df=data.loc[data.symbol==i]
        df=df.loc[df.date>='2017-06-06']
        df=df.loc[df.date<='2018-06-06']
        coins_dict[i] = df.set_index("date").sort_index().loc[:,"volume"]
    df=pd.DataFrame.from_dict(coins_dict).reset_index()
    fig=HorizonPlot(df,"date")
    fig["layout"].update(title="Horizon Graph on Bitcoin Volume")
    iplot(fig)