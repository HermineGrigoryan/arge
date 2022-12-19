import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd

import functions

st.set_page_config(layout="wide", page_title="ARGE-AUA")

st.header('ARGE Business - Time Series Forecasting')

password = st.text_input("Password", type="password")

if password == "aua_arge":

    # Cached functions
    read_and_cache_data = st.cache(pd.read_csv)

    cached_data = read_and_cache_data('arge_df_melted.csv')
    data = cached_data.copy()
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

    st.sidebar.markdown('# User Input Variables')
    navigation = st.sidebar.radio('Navigation', ('Quantitative Analysis', 'Visualizations', 'Models'))


    ##################################
    #### Quantitative Analysis #######
    ##################################

    if navigation == 'Quantitative Analysis':
        st.subheader('Overall Info on the Data')
        with st.expander('Dataset description'):
            '''
            Write here the summary for the project.
            '''
            profiling_table = pd.DataFrame({
                'Number of variables' : [data.shape[1]],
                'Number of observations' : [data.shape[0]],
                'Missing cells' : [data.isna().sum().sum()],
                'Missing cells (%)' : [data.isna().sum().sum()/data.shape[0]]
            })
            st.table(profiling_table)

        st.markdown('_'*100) # adding a breaking line
        st.subheader('Data Exploration')
        head_count = st.slider('How many random rows of data to show?', 5, 50, 5, 5)
        which_columns = st.multiselect('Which columns to show?', data.columns.tolist(), ['item_name', 'arge_group', 'category', 'brand', 'sales', 'month', 'year'])
        st.dataframe(data[which_columns].sample(head_count))

        st.markdown('_'*100) # adding a breaking line
        st.subheader('Summary Statistics of sales per group')
        col1, col2, col3, col4 = st.columns([1,1,1,2])
        grouping_var1 = col1.selectbox('Grouping variable', ['arge_group', 'category', 'brand'])
        grouping_var2 = col2.selectbox('Grouping period', ['year', 'month'])
        continuous_var = col3.selectbox('Continuous variable', ['sales'])
        agg_func = col4.multiselect('Aggregation function', ['mean', 'median', 'std', 'count'], ['mean', 'count'])

        sum_stats = data.groupby([grouping_var1, grouping_var2])[continuous_var].agg(agg_func)
        st.dataframe(sum_stats.reset_index())

    ##################################
    ####### Visualizations ###########
    ##################################
    if navigation == 'Visualizations':
        col1_bar, col2_bar, col3_bar, col4_bar = st.columns([1, 1, 1, 1])
        x_var_bar = col1_bar.selectbox('X variable (barplot)', ['sales'])
        grouping_var_bar = col2_bar.selectbox('Grouping variable (barplot)', 
        ['arge_group', 'category', 'brand'])
        n_obs = col3_bar.slider('Number of items to show', 5, 25, 10, 5)
        agg_func_bar = col4_bar.selectbox('Aggregation function', ['sum', 'mean'])
        st.plotly_chart(functions.top_barplot(data, x_var_bar, 'item_name', grouping_var_bar, n_obs, agg_func_bar), use_container_width=True) 


        # Time series plot
        col1_ts, col2_ts = st.columns([1, 1])
        grouping_var_ts = col1_ts.selectbox('Grouping variable (time series)', 
        ['arge_group', 'category', 'brand'])
        agg_func_ts = col2_ts.selectbox('Aggregation function (time series)', ['sum', 'mean'])
        st.plotly_chart(functions.time_series_plot(data, grouping_var_ts, agg_func_ts), use_container_width=True) 


    if navigation == 'Models':

        # preparing model data
        col1_model, col2_model, col3_model = st.columns([1, 1, 1])
        grouping_var_model = col1_model.selectbox('Grouping variable (for models)', 
        ['arge_group', 'category', 'brand'])
        agg_func_model = col2_model.selectbox('Aggregation function (for models)', ['sum', 'mean'])
        group_df = data.groupby([grouping_var_model, 'date'])['sales'].agg(agg_func_model).reset_index()
        selected_group_model = col3_model.selectbox('Selected group', group_df[grouping_var_model].unique().tolist())
        model_df = group_df[group_df[grouping_var_model] == selected_group_model]

        st.markdown('_'*100) # adding a breaking line
        with st.expander('Moving Average Model'):
            ma_period = st.selectbox('MA period (month)', [2, 3, 4, 5, 6], index=1)
            ma_pred = functions.ma_modeling(model_df, ma_period)
            st.write(f'{ma_period}-month Moving Average for {selected_group_model}')
            st.plotly_chart(functions.pred_vs_actual_plot(ma_pred), use_container_width=True)
            st.write('Accuracy metrics')
            st.dataframe(pd.DataFrame(functions.accuracy_metrics(ma_pred)))
        
        st.markdown('_'*100) # adding a breaking line
        with st.expander('Weighted Moving Average Model'):
            wma_period = st.selectbox('Weighted MA period (month)', [2, 3, 4, 5, 6], index=1)
            cols = st.columns(wma_period)
            weights = []

            for ind, col in enumerate(cols):
                tmp_weight = col.number_input(f'Weight for month {ind+1}', min_value=0.0, max_value=1.0, step=0.1, value=1/wma_period)
                weights.append(tmp_weight) 

            if round(np.sum(weights), 2) != 1.0:
                st.warning(f'Please choose weights in a way, such that their sum is equal to 1! (Current sum={round(np.sum(weights), 2)})')

            wma_pred = functions.wma_modeling(model_df, wma_period, weights)
            st.write(f'{wma_period}-month Weighted Moving Average for {selected_group_model}')
            st.plotly_chart(functions.pred_vs_actual_plot(wma_pred), use_container_width=True)
            st.write('Accuracy metrics')
            st.dataframe(pd.DataFrame(functions.accuracy_metrics(wma_pred)))
        
        st.markdown('_'*100) # adding a breaking line
        with st.expander('Exponential Smoothing Model'):
            alpha = st.slider('Smoothing term (alpha)', 0.1, 1.0, 0.5)
            exp_smoothing_pred = functions.exp_smoothing_model(model_df, alpha)
            st.write(f'Exponential Smoothing (alpha={alpha}) for {selected_group_model}')
            st.plotly_chart(functions.pred_vs_actual_plot(exp_smoothing_pred), use_container_width=True)
            st.write('Accuracy metrics')
            st.dataframe(pd.DataFrame(functions.accuracy_metrics(exp_smoothing_pred)))

        st.markdown('_'*100) # adding a breaking line
        with st.expander('Linear Regression Model'):
            lr_pred = functions.lr_modeling(model_df)
            st.write(f"Linear Regression Model with `month`, `year`, `quarter` and `sales from the previous year's the same month` as independent variables.")
            st.plotly_chart(functions.pred_vs_actual_plot(lr_pred), use_container_width=True)
            st.write('Accuracy metrics')
            st.dataframe(pd.DataFrame(functions.accuracy_metrics(lr_pred)))