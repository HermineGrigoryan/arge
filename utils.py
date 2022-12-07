# It is a good practice to create functions in a separate .py file and 
# import it in the main 'app' file when your code gets big. 

import streamlit as st
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression



def top_barplot(data, x, y, col, n_obs, agg_func):
    grouped_df = data.groupby([y, col])[x].agg([agg_func]).reset_index()
    grouped_df = grouped_df.rename(columns={agg_func: x}) 
    fig = px.bar(grouped_df.sort_values(x, ascending=False).iloc[:n_obs].sort_values(x),
     x = x, y = y, color = col)
    fig.update_layout(xaxis={'categoryorder':'total descending'},
    title = f'Top {n_obs} items with the highest {x} by {agg_func}')
    return fig

def time_series_plot(data, grouping_var, agg_func):

    group_df = data.groupby([grouping_var, 'date'])['sales'].agg(agg_func).reset_index()
    fig = px.line(group_df, 'date', 'sales', color=grouping_var, template = 'plotly_white')

    for i in range(18, 23, 2):
        fig.add_vrect(
            x0=f"20{i}-01-01", x1=f"20{i+1}-01-01",
            fillcolor="#CBE2FB", opacity=0.2,
            layer="below", line_width=0,
        )

        if i < 22:
            fig.add_vrect(
                x0=f"20{i+1}-01-01", x1=f"20{i+2}-01-01",
                fillcolor="#CBFBE3", opacity=0.2,
                layer="below", line_width=0,
            )

    fig.update_traces(visible="legendonly")
    
    return fig


def ma_modeling(model_df, ma_period):
    ma_period = f'{int(ma_period*30)}D'
    model_df['predicted'] = model_df.set_index('date')['sales'].rolling(ma_period).mean().values
    ma_pred = model_df.rename(columns={'sales': 'actual'})

    return ma_pred


def lr_modeling(model_df):
    model_df['lag_sales'] = model_df['sales'].shift(12)
    model_df['year'] = model_df['date'].dt.year
    model_df['quarter'] = (model_df['date'].dt.month -1) // 3 + 1
    model_df['month'] = model_df['date'].dt.month
    model_df = model_df.dropna()
    X = model_df[['year', 'quarter', 'month', 'lag_sales']]
    y = model_df['sales']

    model = LinearRegression().fit(X, y)
    model_df['predicted'] = model.predict(X)

    lr_pred = model_df.rename(columns={'sales': 'actual'})
    return lr_pred



def pred_vs_actual_plot(pred_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['predicted'], name=f'Predicted sales'))
    fig.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['actual'], name=f'Actual sales'))

    return fig
    

def accuracy_metrics(pred_df):
    mad = np.mean(np.abs(pred_df['actual'] - pred_df['predicted']))
    mse = np.mean((pred_df['actual'] - pred_df['predicted'])**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(pred_df['actual'] - pred_df['predicted'])/pred_df['actual'])

    metrics = {'MAD': [mad], 'MSE': [mse], 'RMSE': [rmse], 'MAPE': [mape]}

    return metrics