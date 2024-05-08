# ==============refactor===================
# Use material no for show in clustering.
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import configparser
import urllib
import plotly.express as px
from sqlalchemy import create_engine
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, quantile_transform
from sklearn import metrics
from sklearn.cluster import KMeans,MiniBatchKMeans
from datetime import datetime,timedelta
from datetime import date
import scipy.stats as stat
import information_clustering as ic

# ==================================================================================================

path = '/'.join((os.path.abspath(__file__).replace('\\', '/')).split('/')[:-1])

# @st.cache
def load_data(args_quntile,pertcentile_feature):
    data = pd.read_pickle('./outbound/final_prepare.pkl')
    value_of_quntile = data[pertcentile_feature].quantile(args_quntile)
    data = data.loc[data[pertcentile_feature] <= value_of_quntile]
    return data,value_of_quntile

def clustering(k,df,lst_feature):
    df_clustering = df.drop(columns =['mat_number','Grade','Gram'])
    df_clustering = df[lst_feature]
    X = StandardScaler().fit_transform(df_clustering)
    # k_means = KMeans(n_clusters=k)
    k_means = MiniBatchKMeans(n_clusters=k,random_state=0,batch_size=6)
    model = k_means.fit(X)
    y_hat = k_means.predict(X)
    labels = k_means.labels_

    silhouette_score = metrics.silhouette_score(X, labels, metric = 'euclidean')
    calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels)
    centroids = k_means.cluster_centers_

    # print(silhouette_score)
    # print(calinski_harabasz_score)
    # print(f"Centroids \n{centroids}")

    df_labels = pd.DataFrame(labels, columns = ['Clustering'])
    df['Clustering'] =  df_labels['Clustering']
    return df

def elbow_plot(df,lst_feature):
    df_clustering = df.drop(columns =['mat_number','Grade','Gram'])
    df_clustering = df[lst_feature]
    X = StandardScaler().fit_transform(df_clustering)
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    k_elbow_loop = range(1,20)
    
    for k_elbow in k_elbow_loop:
        # print(k_elbow)
        # kmeanModel = KMeans(n_clusters=k).fit(X)
        k_means_elbow = MiniBatchKMeans(n_clusters=k_elbow,random_state=0,batch_size=6)
        k_means_elbow.fit(X)
 
        distortions.append(sum(np.min(cdist(X, k_means_elbow.cluster_centers_,'euclidean'), axis=1)) / X.shape[0])
        inertias.append(k_means_elbow.inertia_)
    
        mapping1[k_elbow] = sum(np.min(cdist(X, k_means_elbow.cluster_centers_,'euclidean'), axis=1)) / X.shape[0]
        mapping2[k_elbow] = k_means_elbow.inertia_

    # print(type(mapping1))

    df_elbow = pd.DataFrame(list(mapping1.items()),columns = ['Number of K','Distortions']) 

    return df_elbow

def prepare_data(args_start_date,args_end_date):
    # Paramete
    # print(args_start_date)
    # print(args_end_date)
    # args_start_date = '2019-10-01'
    # args_end_date = '2021-09-30'
    # print('-' * 100)
    # print(args_start_date)
    # print(args_end_date)
    # print('-' * 100)
    # start_date = args_start_date
    # end_date = args_end_date
    # base_date = args_end_date

    start_date = datetime.strptime(args_start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args_end_date, '%Y-%m-%d')
    base_date = datetime.strptime(args_end_date, '%Y-%m-%d')

    args_start_date = datetime.strptime(args_start_date, '%Y-%m-%d')
    args_end_date = datetime.strptime(args_end_date, '%Y-%m-%d')

    # start_date = datetime.combine(start_date, datetime.min.time())
    # end_date = datetime.combine(end_date, datetime.min.time())

    # base_date = datetime.strptime(args_end_date, '%Y-%m-%d')
    print('*' * 100)
    print(end_date)
    print(start_date)
    print('*' * 100)

    diff_date = end_date - start_date
    # number_of_week = round(diff_date.days / 7,2)
    number_of_month = round(diff_date.days / 30,0)

    # print("Number of month : " ,number_of_month)

    # ========================================================================================
    get_demands(args_start_date,args_end_date)
    demands_df = pd.read_pickle(os.path.join(path, './temp/demands.pkl'))
    # demands_df['dp_date'] = datetime.date.fromisoformat(demands_df.dp_date)
    # demands_df['dp_date'] = demands_df['dp_date'].to_pydatetime()
    demands_df['dp_date'] = pd.to_datetime(demands_df['dp_date'], dayfirst=True)
    demands_df_filter = (demands_df['dp_date'] <= args_end_date)
    demands_df = demands_df.loc[demands_df_filter]

    lst_df_stat_calculation = ['mat_number','Grade','Gram','dp_date','ton']
    demand_stat_calculation_df_source = demands_df[lst_df_stat_calculation]
    demand_stat_calculation_df_source = demand_stat_calculation_df_source[~((demand_stat_calculation_df_source['dp_date'].dt.strftime('%m-%d') >= '04-13') & (demand_stat_calculation_df_source['dp_date'].dt.strftime('%m-%d')  <= '04-15'))]
    # demand_stat_calculation_df_source = demand_stat_calculation_df_source[demands_df['dp_date'].dt.month]

    demand_stat_calculation_df_source['month_number'] = demands_df['dp_date'].dt.month
    demand_stat_calculation_df_source['week_number'] = demands_df['dp_date'].apply(lambda x: x.isocalendar()[1])
    demand_stat_calculation_df_source['year'] = demands_df['dp_date'].dt.year
    
    # print(demand_stat_calculation_df_source)
    demands_df.to_excel('intern_source.xlsx',index=False)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # User define : 05/05/2022 (New Requirement)
    # Filter out data from  : Songkran (13,14,15) 
    # Choose Except : Year Month (exclude)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Year = 2021 Week = 3 => 2021_3 Cross 

    demand_stat_calculation_df_source = demand_stat_calculation_df_source.sort_values(['mat_number','Grade','Gram','dp_date'])

    # ========================================================================================
    # monthly
    lst_df_stat_calculation = ['mat_number','Grade','Gram','month_number','ton']
    
    # print('=' * 100)
    summary_monthly_demand_stat_calculation_df = demand_stat_calculation_df_source[lst_df_stat_calculation]
    # print(summary_monthly_demand_stat_calculation_df.loc[summary_monthly_demand_stat_calculation_df['mat_number'] == 'Z02WS-140D1740117N'])
    summary_monthly_demand_stat_calculation_df = summary_monthly_demand_stat_calculation_df.groupby(['mat_number','Grade','Gram','month_number']).sum().reset_index()
    summary_monthly_demand_stat_calculation_df.columns = summary_monthly_demand_stat_calculation_df.columns.str.replace('ton', 'summary_monthly_sales_volume')
    # print(summary_monthly_demand_stat_calculation_df.loc[summary_monthly_demand_stat_calculation_df['mat_number'] == 'Z02WS-140D1740117N'])

    average_monthly_demand_stat_calculation_df = summary_monthly_demand_stat_calculation_df[['mat_number','Grade','Gram','summary_monthly_sales_volume']]
    average_monthly_demand_stat_calculation_df = average_monthly_demand_stat_calculation_df.groupby(['mat_number','Grade','Gram']).mean().reset_index()
    average_monthly_demand_stat_calculation_df.columns = average_monthly_demand_stat_calculation_df.columns.str.replace('summary_monthly_sales_volume', 'average_monthly_sales_volume')
    # print(average_monthly_demand_stat_calculation_df.loc[average_monthly_demand_stat_calculation_df['mat_number'] == 'Z02WS-140D1740117N'])

    sd_monthly_demand_stat_calculation_df = summary_monthly_demand_stat_calculation_df[['mat_number','Grade','Gram','summary_monthly_sales_volume']]
    sd_monthly_demand_stat_calculation_df = sd_monthly_demand_stat_calculation_df.groupby(['mat_number','Grade','Gram']).std().reset_index()
    sd_monthly_demand_stat_calculation_df.columns = sd_monthly_demand_stat_calculation_df.columns.str.replace('summary_monthly_sales_volume', 'sd_monthly_sales_volume')
    # print(sd_monthly_demand_stat_calculation_df.loc[sd_monthly_demand_stat_calculation_df['mat_number'] == 'Z02WS-140D1740117N'])

    stat_monthly_merge_df = pd.merge(average_monthly_demand_stat_calculation_df, sd_monthly_demand_stat_calculation_df)
    stat_monthly_merge_df['cv_monthly'] = stat_monthly_merge_df['sd_monthly_sales_volume']/stat_monthly_merge_df['average_monthly_sales_volume']
    # print('=' * 100)
    
    # ========================================================================================
    # weekly
    lst_df_stat_calculation = ['mat_number','Grade','Gram','week_number','ton']
    summary_weekly_demand_stat_calculation_df = demand_stat_calculation_df_source[lst_df_stat_calculation]
    summary_weekly_demand_stat_calculation_df = summary_weekly_demand_stat_calculation_df.groupby(['mat_number','Grade','Gram','week_number']).sum().reset_index()
    summary_weekly_demand_stat_calculation_df.columns = summary_weekly_demand_stat_calculation_df.columns.str.replace('ton', 'summary_weekly_sales_volume')
    # print(summary_weekly_demand_stat_calculation_df)

    average_weekly_demand_stat_calculation_df = summary_weekly_demand_stat_calculation_df[['mat_number','Grade','Gram','summary_weekly_sales_volume']]
    average_weekly_demand_stat_calculation_df = average_weekly_demand_stat_calculation_df.groupby(['mat_number','Grade','Gram']).mean().reset_index()
    average_weekly_demand_stat_calculation_df.columns = average_weekly_demand_stat_calculation_df.columns.str.replace('summary_weekly_sales_volume', 'average_weekly_sales_volume')
    # print(average_weekly_demand_stat_calculation_df)

    sd_weekly_demand_stat_calculation_df = summary_weekly_demand_stat_calculation_df[['mat_number','Grade','Gram','summary_weekly_sales_volume']]
    sd_weekly_demand_stat_calculation_df = sd_weekly_demand_stat_calculation_df.groupby(['mat_number','Grade','Gram']).std().reset_index()
    sd_weekly_demand_stat_calculation_df.columns = sd_weekly_demand_stat_calculation_df.columns.str.replace('summary_weekly_sales_volume', 'sd_weekly_sales_volume')
    # print(sd_weekly_demand_stat_calculation_df)

    stat_weekly_merge_df = pd.merge(average_weekly_demand_stat_calculation_df, sd_weekly_demand_stat_calculation_df)
    stat_weekly_merge_df['cv_weekly'] = stat_weekly_merge_df['sd_weekly_sales_volume']/stat_weekly_merge_df['average_weekly_sales_volume']
    stat_weekly_merge_df.to_pickle(os.path.join(path, './temp/stat_weekly.pkl'))
    # print(stat_weekly_merge_df)
    # ========================================================================================

    lst_df_sales_frequency = ['mat_number','Grade','Gram','dp_date']
    sales_frequency_df = demands_df[lst_df_sales_frequency]
    sales_frequency_df = sales_frequency_df.drop_duplicates()
    sales_frequency_df = sales_frequency_df.sort_values(['mat_number','Grade','Gram'])

    # dp_date_filter = (sales_frequency_df['dp_date'] >= args_start_date) & (sales_frequency_df['dp_date'] <= args_end_date)
    # sales_frequency_df = sales_frequency_df.loc[dp_date_filter]
    sales_frequency_df.to_pickle(os.path.join(path, './temp/demands_sales_frequency.pkl'))
    sales_frequency_df = pd.read_pickle(os.path.join(path, './temp/demands_sales_frequency.pkl'))

    # with pd.ExcelWriter("./example/sales_frequency_df.xlsx", engine='openpyxl') as writer:

        # demand_stat_calculation_df_source[['Grade','Gram','dp_date','ton']].to_excel(writer,index=False,sheet_name='raw_data')

    demand_stat_calculation_df_source[['mat_number','Grade','Gram','dp_date','ton']].to_pickle(os.path.join(path, './temp/raw_demands.pkl'))

    sales_frequency_df['month'] = sales_frequency_df['dp_date'].dt.month
    sales_frequency_df['year'] = sales_frequency_df['dp_date'].dt.year
    # sales_frequency_df.to_excel(writer,index=False,sheet_name='sales_frequency_raw_data')
    # sales_frequency_df = sales_frequency_df.reset_index()
    # print(sales_frequency_df)
    # sales_frequency_df.to_excel('sales_frequency_df_mont_year.xlsx',index=False)
    
    # print(sales_frequency_df)
    print('-' * 100)
    print(number_of_month)
    print('-' * 100)

    sales_frequency_df = sales_frequency_df[['mat_number','Grade','Gram','year','month']]
    sales_frequency_df = sales_frequency_df.drop_duplicates()
    sales_frequency_df = sales_frequency_df[['mat_number','Grade','Gram','month']]
    sales_frequency_df = sales_frequency_df.groupby(['mat_number','Grade','Gram']).count()
    sales_frequency_df.rename(columns={'month':'number_of_month'}, inplace=True)
    sales_frequency_df['diff_month'] = number_of_month - sales_frequency_df['number_of_month']

    # print(sales_frequency_df[['diff_month','number_of_month']].loc[sales_frequency_df['diff_month'] <= 12])
    # sales_frequency_df.to_excel('sales_frequency_df.xlsx',index = False)
    conditions = [
        (sales_frequency_df['diff_month']  >= 0) & (sales_frequency_df['diff_month'] <= 3),
        (sales_frequency_df['diff_month']  >= 4) & (sales_frequency_df['diff_month'] <= 6),
        (sales_frequency_df['diff_month']  >= 7) & (sales_frequency_df['diff_month'] <= 12),
        (sales_frequency_df['diff_month']  > 12)
    ]
    choices = ['Always', 'Usually', 'Seldom','Dead Stock']
    sales_frequency_df['sales_frequency'] = np.select(conditions, choices)
    # sales_frequency_df.to_excel('sales_frequency_df.xlsx')

    # --------------------------------------------------------------------------------------------------------------
    # Find consecutive count
    lst_df_consecutive_count = ['mat_number','Grade','Gram','dp_date']
    cols = ['mat_number','Grade','Gram']
    # consecutive_count_df = demands_df[lst_df_consecutive_count].loc[demands_df['mat_number'] == 'Z02CA-090D0630117N']
    consecutive_count_df = demands_df[lst_df_consecutive_count]
    # print(consecutive_count_df)
    consecutive_count_df['month'] = consecutive_count_df['dp_date'].dt.month
    consecutive_count_df['year'] = consecutive_count_df['dp_date'].dt.year
    # consecutive_count_df['year'] = 0
    # ==========================================================================

    consecutive_count_df = consecutive_count_df[['mat_number','Grade','Gram','month','year']].drop_duplicates()
    # consecutive_count_df = consecutive_count_df.reset_index()
    consecutive_count_df = consecutive_count_df.sort_values(['mat_number','Grade','Gram','year','month'])
    # consecutive_count_df.to_excel('consecutive_count_df.xlsx')
    # print('*' * 100)
    # print(consecutive_count_df)
    g = consecutive_count_df.groupby(cols)
    month_diff = g['month'].diff()
    year_diff = g['year'].diff()
    # year_month_diff = (year_diff * 12) + month_diff
    
    nonconsecutive = ~((year_diff.eq(0) & month_diff.eq(1)) | (year_diff.eq(1) & month_diff.eq(-11)))
    # nonconsecutive = ()

    # nonconsecutive = ~((year_diff.eq(0) & month_diff.eq(1)) | (year_diff.eq(1) & month_diff.eq(-2)))
    consecutive_count_df = consecutive_count_df.groupby([*cols, nonconsecutive.cumsum()]).size().droplevel(-1).groupby(cols).max().reset_index(name='max_consecutive_month')
    # print(consecutive_count_df)

    # consecutive_count_df['month_diff'] = consecutive_count_df['number_of_month'].diff()
    # year_diff = consecutive_count_df['year'].diff()

    # print(consecutive_count_df)
    # nonconsecutive = ~((year_diff.eq(0) & month_diff.eq(1)) | (year_diff.eq(1) & month_diff.eq(-9)))
    # out = consecutive_count_df.groupby([*cols, nonconsecutive.cumsum()]).size().droplevel(-1).groupby(cols).max().reset_index(name='counts')

    # print(out)
    # --------------------------------------------------------------------------------------------------------------

    # sales_frequency_df.to_excel(writer,index=False,sheet_name='sales_frequency')
        # print(sales_frequency_df)

        # stat_monthly_merge_df.to_excel(writer,index=False,sheet_name='statistic_monthly')
        # stat_weekly_merge_df.to_excel(writer,index=False,sheet_name='statistic_weekly')

    ans_df = pd.merge(stat_monthly_merge_df, stat_weekly_merge_df)
    ans_df = ans_df.fillna(0)
    # ans_df.to_excel('ans_df_before_join.xlsx')
        # ans_df.to_excel(writer,index=False,sheet_name='final_statistic')
    ans_df = pd.merge(ans_df, sales_frequency_df, how='left', on=['mat_number','Grade','Gram'])
    ans_df = pd.merge(ans_df, consecutive_count_df, how='left', on=['mat_number','Grade','Gram'])

    # ans_df['sales_frequency'] = np.where((ans_df['sales_frequency'] == 'Usually') & (ans_df['max_consecutive_month'] > 6) , 'Usually :High Potential',
    #      np.where((ans_df['sales_frequency'] == 'Usually') & ((ans_df['max_consecutive_month'] > 4) & (ans_df['max_consecutive_month'] <= 6)) , 'Usually :Low Potential',
    #      np.where((ans_df['sales_frequency'] == 'Usually') & (ans_df['max_consecutive_month'] <= 4) , 'Usually :Expect to Terminate', ans_df['sales_frequency'] 
    # )))

    ans_df['sales_frequency'] = np.where((ans_df['sales_frequency'] == 'Usually') & (ans_df['max_consecutive_month'] >= (number_of_month * 0.30) ) , 'Usually :High Potential',
         np.where((ans_df['sales_frequency'] == 'Usually') & ((ans_df['max_consecutive_month'] > (number_of_month * 0.25)) & (ans_df['max_consecutive_month'] < (number_of_month * 0.30))) , 'Usually :Low Potential',
         np.where((ans_df['sales_frequency'] == 'Usually') & (ans_df['max_consecutive_month'] <= (number_of_month * 0.25)) , 'Usually :Expect to Terminate', ans_df['sales_frequency'] 
    )))
    
    print(ans_df)

    ans_df['sales_frequency'] = ans_df['sales_frequency']. fillna('Dead Stock')
    # ans_df.to_pickle(os.path.join(path, './outbound/final_prepare.pkl'))
    ans_df.to_excel('ans_df_new.xlsx')
    check_grouping(demands_df,start_date,end_date)

# -------------------------------------------------------------------------------------------
def get_demands(args_start_date,args_end_date):
    config = configparser.ConfigParser()
    config.read(os.path.join(path, 'initial.ini'))
    dest_server = config['AI_Demand']['server'] 
    dest_database = config['AI_Demand']['database']
    dest_username = config['AI_Demand']['username']
    dest_password = config['AI_Demand']['password']
    conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+dest_server+';DATABASE='+dest_database+';UID='+dest_username+';PWD='+ dest_password
    params = urllib.parse.quote_plus(conn_str)
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
    demands_query = open(os.path.join(path, '.\Sql_scripts\demands.sql'), 'r').read()
    demands_query = demands_query %(args_start_date,args_start_date,args_start_date,args_end_date)
    demands_df = pd.read_sql_query(demands_query,con=engine)
    demands_df['Grade'] = demands_df['Grade'].str.strip()

    demands_df.to_pickle(os.path.join(path, './temp/demands.pkl'))
    return demands_df

def get_cogs(args_condition):
    config = configparser.ConfigParser()
    config.read(os.path.join(path, 'initial.ini'))
    dest_server = config['COGS']['server'] 
    dest_database = config['COGS']['database']
    dest_username = config['COGS']['username']
    dest_password = config['COGS']['password']
    conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+dest_server+';DATABASE='+dest_database+';UID='+dest_username+';PWD='+ dest_password
    params = urllib.parse.quote_plus(conn_str)
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
    cogs_query = open(os.path.join(path, '.\Sql_scripts\cogs.sql'), 'r').read()
    cogs_query = cogs_query %(args_condition,args_condition)
    cogs_df = pd.read_sql_query(cogs_query,con=engine)
    return cogs_df

def get_rwds(args_condition):
    config = configparser.ConfigParser()
    config.read(os.path.join(path, 'initial.ini'))
    dest_server = config['COGS']['server'] 
    dest_database = config['COGS']['database']
    dest_username = config['COGS']['username']
    dest_password = config['COGS']['password']
    conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+dest_server+';DATABASE='+dest_database+';UID='+dest_username+';PWD='+ dest_password
    params = urllib.parse.quote_plus(conn_str)
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
    rwds_query = open(os.path.join(path, '.\Sql_scripts\sql_rwds.sql'), 'r').read()
    rwds_query = rwds_query %(args_condition,args_condition)
    rwds_df = pd.read_sql_query(rwds_query,con=engine)
    return rwds_df 

def get_leadtime(args_condition):
    config = configparser.ConfigParser()
    config.read(os.path.join(path, 'initial.ini'))
    dest_server = config['COGS']['server'] 
    dest_database = config['COGS']['database']
    dest_username = config['COGS']['username']
    dest_password = config['COGS']['password']
    conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+dest_server+';DATABASE='+dest_database+';UID='+dest_username+';PWD='+ dest_password
    params = urllib.parse.quote_plus(conn_str)
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
    leadtime_query = open(os.path.join(path, '.\Sql_scripts\production_leadtime.sql'), 'r').read()
    leadtime_query = leadtime_query %(args_condition)
    leadtime_df = pd.read_sql_query(leadtime_query,con=engine)
    return leadtime_df

# ==================================================================================================
def check_grouping(args_df_demand,args_start_dt,args_end_dt):

    start_date = args_start_dt
    end_date = args_end_dt

    end_month = args_end_dt.month
    end_year = args_end_dt.year

    sales_frequency_df = args_df_demand.copy()
    selected_column = ['mat_number','dp_date','ton','Grade','Gram']
    sales_frequency_df = sales_frequency_df[selected_column]
    number_of_days = sales_frequency_df['dp_date'].nunique()

    sales_frequency_df = sales_frequency_df.rename(
    columns={'mat_number': 'Mat Number'}
    )

    # sales_frequency_df = sales_frequency_df.resample('M', on='dp_date').interpolate()
    sales_frequency_df.dp_date = pd.to_datetime(sales_frequency_df.dp_date)
    sales_frequency_df['month'] = sales_frequency_df['dp_date'].dt.month
    sales_frequency_df['year'] = sales_frequency_df['dp_date'].dt.year
    # min_date = sales_frequency_df['dp_date'].min() + 1
    sales_frequency_df['week_number'] = sales_frequency_df['dp_date'].dt.isocalendar().week
    sales_frequency_df['month_number'] = sales_frequency_df['year'].map(str) + sales_frequency_df['month'].map(str)
    
    # # =================================================================================================
    daily_df = sales_frequency_df[['Mat Number','Grade','Gram','ton']]

    daily_df = daily_df.groupby(['Mat Number','Grade','Gram']).agg(
        daily_std=('ton','std'),
        daily_avg=('ton','mean')
    )
    # =================================================================================================
    weekly_df = sales_frequency_df[['Mat Number','Grade','Gram','week_number','ton']]
    weekly_df = weekly_df.groupby(['Mat Number','Grade','Gram','week_number']).agg(
        summary_ton_weekly=('ton','sum'),
    )

    weekly_df = weekly_df.groupby(['Mat Number','Grade','Gram']).agg(
        weekly_std=('summary_ton_weekly','std'),
        weekly_avg=('summary_ton_weekly','mean')
    )

    weekly_df.to_excel('./debug/01_weekly_df.xlsx')

    # =================================================================================================
    monthly_df = sales_frequency_df[['Mat Number','Grade','Gram','month_number','ton']]
    monthly_df = monthly_df.groupby(['Mat Number','Grade','Gram','month_number']).agg(
        summary_ton_monthly=('ton','sum'),
    )

    monthly_df = monthly_df.groupby(['Mat Number','Grade','Gram']).agg(
        monthly_std=('summary_ton_monthly','std'),
        monthly_avg=('summary_ton_monthly','mean')
    )

    monthly_df.to_excel('./debug/01_monthly_df.xlsx')

    # =================================================================================================

    # print(sales_frequency_df)
    monday1 = (start_date - timedelta(days=start_date.weekday()))
    monday2 = (end_date - timedelta(days=end_date.weekday()))

    number_of_month = ((end_date.year - start_date.year) * 12 + end_date.month - start_date.month) + 1
    number_of_week = round((monday2 - monday1).days / 7,2)
    print('*' * 100)
    print(number_of_month)
    print(number_of_week)
    print('*' * 100)

    main_df_for_ton = sales_frequency_df[['Mat Number','ton']]
    map_grade_gram = sales_frequency_df[['Mat Number','Grade','Gram']]
    map_grade_gram = map_grade_gram.drop_duplicates()

    main_df_for_ton = main_df_for_ton.groupby(['Mat Number']).agg(
        summary_ton=('ton','sum'),
        )

    # === Check Dead Stock ===
    selected_column = ['Mat Number','month','year']
    sales_frequency_df = sales_frequency_df[selected_column]
    sales_frequency_df = sales_frequency_df.drop_duplicates()
    sales_frequency_df['sale_flag'] = 1

    time_df = pd.DataFrame.from_dict({'dp_date':pd.date_range(start=start_date, end=end_date,freq='M')})
    time_df['key'] = 1
    # print(time_df)

    sales_frequency_df_dummy = sales_frequency_df[['Mat Number']].copy()
    sales_frequency_df_dummy['key'] = 1

    sales_frequency_df_dummy = pd.merge(sales_frequency_df_dummy, time_df, on ='key')
    sales_frequency_df_dummy = sales_frequency_df_dummy.drop(columns='key',axis=1)
    sales_frequency_df_dummy['month'] = sales_frequency_df_dummy['dp_date'].dt.month
    sales_frequency_df_dummy['year'] = sales_frequency_df_dummy['dp_date'].dt.year
    sales_frequency_df_dummy = sales_frequency_df_dummy[selected_column]
    sales_frequency_df_dummy = sales_frequency_df_dummy.drop_duplicates()
    sales_frequency_df_dummy['sale_flag'] = 0

    # sales_frequency_df = sales_frequency_df.groupby(['Mat Number']).count()
    # print(sales_frequency_df_dummy)

    prepare_df = pd.concat([sales_frequency_df, sales_frequency_df_dummy])
    prepare_df = prepare_df.groupby(['Mat Number','month','year']).sum().reset_index()
    prepare_df = prepare_df.sort_values(by = ['Mat Number','year','month'])
    # prepare_df.to_excel('./outbound/prepare_df.xlsx',index=False)
    
    previous_month_sales = 0
    max_continuous_non_sales = 0
    current_max_continuous_non_sales = 0
    previous_mat = ''

    for index, row in prepare_df.iterrows():
        if (previous_mat == row['Mat Number']) or (previous_mat == '') :
            if (row['sale_flag'] == 0):
                current_max_continuous_non_sales += 1

            if ((row['sale_flag'] == 1) or (row['year'] == end_year and row['month'] == end_month)):
                if current_max_continuous_non_sales > max_continuous_non_sales:
                    max_continuous_non_sales =  current_max_continuous_non_sales

                current_max_continuous_non_sales = 0
            previous_mat = row['Mat Number']
            previous_month_sales = row['sale_flag']
        else:

            # print(f'Maximum continuous non sales month of {previous_mat} : {max_continuous_non_sales}')
            prepare_df.loc[(prepare_df['Mat Number'] == previous_mat), 'max_non_sales'] = max_continuous_non_sales
            previous_mat = row['Mat Number']
            current_max_continuous_non_sales = 0
            previous_month_sales = 0
            max_continuous_non_sales = 0
            current_max_continuous_non_sales = 0

            if row['sale_flag'] == 0:
                current_max_continuous_non_sales += 1
    
    if current_max_continuous_non_sales > max_continuous_non_sales:
        max_continuous_non_sales =  current_max_continuous_non_sales
    print(f'Maximum no sales continuous of {previous_mat} : {max_continuous_non_sales}')
    prepare_df.loc[(prepare_df['Mat Number'] == previous_mat), 'max_non_sales'] = max_continuous_non_sales

    select_column = ['Mat Number','max_non_sales']
    prepare_df = prepare_df[select_column]
    prepare_df = prepare_df.drop_duplicates()
    # prepare_df.to_excel('./outbound/final.xlsx',index=False)

    conditions = [
            (prepare_df['max_non_sales']  >= 0) & (prepare_df['max_non_sales'] <= 3),
            (prepare_df['max_non_sales']  == 4),
            (prepare_df['max_non_sales'] == 5),
            (prepare_df['max_non_sales'] == 6),
            (prepare_df['max_non_sales']  >= 7) & (prepare_df['max_non_sales'] <= 12),
            (prepare_df['max_non_sales']  > 12)
        ]
    choices = [
        'Always', 
        'Usually :High Potential',
        'Usually :Low Potential',
        'Usually :Expect to Terminate',
        'Seldom',
        'Dead Stock'
    ]
    prepare_df['sales_frequency'] = np.select(conditions, choices)

    result_summary_df = main_df_for_ton.merge(prepare_df,how='left',on=['Mat Number'])
    result_summary_df = result_summary_df.merge(map_grade_gram,how='left',on=['Mat Number'])
    result_summary_df = result_summary_df.merge(weekly_df,how='left',on=['Mat Number'])
    result_summary_df = result_summary_df.merge(monthly_df,how='left',on=['Mat Number'])
    # result_summary_df = result_summary_df.merge(daily_df,how='left',on=['Mat Number'])

    result_summary_df.to_excel('result_summary_df.xlsx',index=False)

    result_summary_df = result_summary_df.drop_duplicates()
    result_summary_detail_df = result_summary_df.copy()
    result_summary_detail_df.to_excel('result_summary_detail_df.xlsx')

    result_summary_df = result_summary_df[['Mat Number','Grade','Gram','sales_frequency','summary_ton','weekly_std','weekly_avg','monthly_std']]
    result_summary_df = result_summary_df.groupby(['sales_frequency']).agg(
        number_of_sku =('sales_frequency','count'),
        total_ton = ('summary_ton','sum'),
        # std_weekly = ('weekly_std','mean'),
        # avg_weekly = ('weekly_avg','mean'),
        # std_monthly = ('monthly_std','mean')
        std_monthly=('monthly_std', lambda x: x.std() if len(x) >= 2 else x),
        std_weekly=('weekly_std', lambda x: x.std() if len(x) >= 2 else x),
        )
    
    result_summary_df['avg_weekly'] = result_summary_df['total_ton'] / number_of_week
    result_summary_df['avg_monthly'] = result_summary_df['total_ton'] / number_of_month
    result_summary_df['weekly_cv'] = result_summary_df['std_weekly'] / result_summary_df['avg_weekly'] 
    result_summary_df['monthly_cv'] = result_summary_df['std_monthly'] / result_summary_df['avg_monthly'] 

    result_summary_detail_df = result_summary_detail_df.merge(daily_df,how='left',on=['Mat Number'])

    result_summary_detail_df = result_summary_detail_df[['Mat Number','Grade','Gram','sales_frequency','summary_ton','weekly_std','weekly_avg','monthly_std','daily_avg','daily_std']]
    result_summary_detail_df = result_summary_detail_df.groupby(['Mat Number','Grade','Gram','sales_frequency']).agg(
        number_of_sku=('sales_frequency','count'),
        total_ton = ('summary_ton','sum'),
        # std_weekly = ('weekly_std','mean'),
        std_monthly=('monthly_std', lambda x: x.std() if len(x) >= 2 else x),
        std_weekly=('weekly_std', lambda x: x.std() if len(x) >= 2 else x),
       
        # std_monthly = ('monthly_std','mean'), 
        avg_daily = ('daily_avg','mean'),
        std_daily = ('daily_std','mean')
        ).reset_index()
    print("result_summary_detail_df")
 
    # Replace NaN values with 0 in 'std_monthly' and 'std_weekly' columns
    result_summary_detail_df['std_monthly'].fillna(0, inplace=True)
    result_summary_detail_df['std_weekly'].fillna(0, inplace=True)
    result_summary_detail_df['avg_weekly'] = result_summary_detail_df['total_ton'] / number_of_week
    result_summary_detail_df['avg_daily'] = result_summary_detail_df['total_ton'] / number_of_days
    result_summary_detail_df['avg_monthly'] = result_summary_detail_df['total_ton'] / number_of_month
    result_summary_detail_df['weekly_cv'] = result_summary_detail_df['std_weekly'] / result_summary_detail_df['avg_weekly'] 
    result_summary_detail_df['monthly_cv'] = result_summary_detail_df['std_monthly'] / result_summary_detail_df['avg_monthly'] 

 
    
    result_summary_detail_df = result_summary_detail_df.rename(
    columns={
        'Mat Number': 'mat_number',
        'std_weekly':'sd_weekly_sales_volume',
        'avg_weekly':'average_weekly_sales_volume',
        'weekly_cv' : 'cv_weekly',
        'avg_monthly': 'average_monthly_sales_volume',
        'monthly_cv' : 'cv_monthly',
        'std_monthly': 'sd_monthly_sales_volume',
        }
    )

    
    print(result_summary_detail_df)

    result_summary_df.to_pickle(os.path.join(path, './outbound/result_summary_df.pkl'))
    result_summary_detail_df.to_pickle(os.path.join(path, './outbound/final_prepare.pkl'))

    # print(result_summary_df)
    # print(result_summary_detail_df)
    result_summary_detail_df.to_excel('./debug/01_result_summary_detail_df.xlsx',index = False)
    result_summary_df.to_excel('./debug/02_result_summary_df.xlsx')

# ==================================================================================================
# start_period = '2021-10-01'
# end_period = '2022-09-30'
# ==================================================================================================
# =================Presentation Layer=================
if __name__ == '__main__':

    start_period = datetime.strptime('2021-10-01', '%Y-%m-%d')
    end_period = datetime.strptime('2022-09-30', '%Y-%m-%d')
    str_start_period = start_period.strftime('%Y-%m-%d')
    str_end_period = end_period.strftime('%Y-%m-%d')

    lst_of_service_level = [60,65,70,75,80,85,90,95]
    # clustering_obj = ic.Clustering()
    # df_inf_clustering = clustering_obj.get_inf_decision(str_start_period,str_end_period)

    # prepare_data(str_start_period,str_end_period)

    # df_inf_clustering.to_excel('df_inf_clustering.xlsx', index=False)
    # st.session_state.df_choose = df_choose
    # tbl_choose = st.table(df_choose)
    # df_lt = clustering_obj.get_leadtime(lt_year)
    df_choose_lst = [
        'inventory_turnover',
        'ratio (STD:LOW:NON)',
        'ratio (STD:LOW:NON) sales volumn',
        'inventory_days','avg_inventory',
        'revise_no_of_std_non_std',
        'main_grade'
    ]

    df_choose = pd.DataFrame(columns=df_choose_lst)
    cogs_wip_raw_mat = 2856.00
    coe_domestic = 0.75
    
    # Decision Params
    percentile = 95
    weekly_cv = 0.6
    monthly_cv = 0.6

    cv_lower_bound = 0.4829
    cv_uper_bound = 0.8307
    avg_monthly_uper_bound = 79.0
    avg_monthly_lower_bound =40.5
    

    percentile_sales_vol_upper = 67
    percentile_sales_vol_lower = 33
    percentile_cv_upper = 15
    percentile_cv_lower = 50
    number_of_k = 3
    service_level = 93.00
    
    first_feature = ''
    second_feature = 'cv_weekly'
    pertcentile_feature = ''
    quantile = percentile / 100
    wacc = 12
    holding_cost = 0.00
    operating_cost = 0
    net_sales = 0
    current_year = date.today().year
    year_leadtime = current_year - 1
    cogs = 22100.00
    holding_cost = 160.0
    wacc = 11.5

    lst_period = [i for i in range(current_year - 5,current_year + 1)]
    lst_month = ['',1,2,3,4,5,6,7,8,9,10,11,12]
    lst_period_month = []
    lst_exclude_period_cogs = []
    lst_exclude_period_rwds = []
    i = 1
    lst_str_cogs = None
    lst_str_rwds = None

    for period in lst_period:
        for month in lst_month:
            add_period = str(period) + '_' + str(month)
            lst_period_month.append(add_period)

    with st.form("input_params_form"):    
                  
        
        # with st.sidebar:
        #     st.title("Condition")
        #     start_period = st.date_input(label="Start period",value=datetime.strptime(start_period, '%Y/%m/%d'))
        #     end_period = st.date_input(label="End period",value=datetime.strptime(end_period, '%Y/%m/%d'))
        #     number_of_k = st.number_input(label="Choose number of cluster",value=number_of_k,step=1,format='%i')           
        # prepare_data(start_period,end_period)
        #     src_df,quantile_value = load_data(quantile,pertcentile_feature)
        #     lst_columns_name = list(src_df.columns)
        #     ans_df = pd.DataFrame
        #     lst_columns_name = lst_columns_name[2:]

        #     pertcentile_feature = st.selectbox("Feature for calculate percentile : ", options = lst_columns_name,index=5)
            
        #     quantile = percentile / 100
        #     src_df,quantile_value = load_data(quantile,pertcentile_feature)
        #     st.write(f"Percentile value : {quantile_value}")

        #     # range_cv = st.slider("Coefficient of Variation",value=(0.00, 1.00),step=0.01)
        #     # npd_data = st.file_uploader("Import File (New product development and product phase out)",help="""Column List : Maiiai""")
             
        #     first_feature = st.selectbox("1st feature", options = lst_columns_name,index=1)
        #     second_feature = st.selectbox("2nd feature", options = lst_columns_name,index=5)
        
        with st.sidebar:
            st.title("Parameters for decision")

            start_period = st.date_input("Data date from :",value=start_period)
            end_period = st.date_input("Data date to :",value=end_period)
            str_start_period = start_period.strftime('%Y-%m-%d')
            str_end_period = end_period.strftime('%Y-%m-%d')

            # print('=' * 100)
            # print(start_period)
            # print(end_period)
            # print(str_start_period)
            # print(str_end_period)
            # print('=' * 100)
            prepare_data(str_start_period,str_end_period) 
            clustering_obj = ic.Clustering()
            df_inf_clustering = clustering_obj.get_inf_decision(str_start_period,str_end_period)

            # prepare_data(str_start_period,str_end_period)

            percentile = st.number_input(label=f"Percentile of sales volume (%)",value=percentile,min_value=0,max_value=100,step=1)
            quantile = percentile / 100

            # monthly_cv = st.number_input(label="Monthly CV",value = monthly_cv ,step=0.1)
            weekly_cv = st.number_input(label="Weekly CV",value = weekly_cv ,step=0.1)

            src_df,quantile_value,src_sales_freq_df,clustering_df = clustering_obj.set_std(quantile,weekly_cv)
            src_sales_freq_df.to_excel('./debug/04-Sales-Frequency.xlsx', index=False, engine='openpyxl')
            # src_sales_freq_df.to_excel('src_sales_freq_df.xlsx',index=False)
        
            number_of_k = st.number_input(label="Choose number of cluster",value=number_of_k,step=1,format='%i')
            year_leadtime = st.selectbox('Leadtime Period ',lst_period,index= len(lst_period) - 2 )
            service_level = st.number_input(label=f"Percent Of Service Level (%)",value=service_level,min_value=0.00,max_value=100.00,step=1.00)
            z_score = stat.norm.ppf(service_level/100)
            cogs = st.number_input(label=f"Cost of goods sold FG (million THB)",value=cogs,min_value=0.00,max_value=10000000.00,step=1.00)

            cogs_wip_raw_mat = st.number_input(label=f"Cost of goods sold WIP & Raw material (million THB)",value=cogs_wip_raw_mat,min_value=0.00,max_value=10000000.00,step=1.00)

            coe_domestic = st.number_input(label=f"Domestic Portion",value=coe_domestic,min_value=0.00,max_value=1.00,step=0.01)
            # st.write(str(z_score))
            # year_leadtime = st.number_input(label=f"Lead time Year",value=year_leadtime,min_value=0,max_value=9999,step=1)
            # print("=" * 100)
            # clustering_df.to_excel('clustering_df.xlsx',index=False)
            # print("=" * 100)
            # clustering_df = clustering_df[]

            # == Eliminate outlier ==
            # clustering_df = clustering_df.loc[clustering_df['avg_monthly'] <= quantile_value]

            return_clustring_df,df_elbow,df_centroid,percentile_df = clustering_obj.clustering(clustering_df,number_of_k)

            st.title("Parameters for clustering")
            avg_monthly_uper_bound = st.number_input(label=f"Sales volume upper bound",value=avg_monthly_uper_bound,min_value=0.00,max_value=1000.00,step=0.1) 
            avg_monthly_lower_bound = st.number_input(label=f"Sales volume lower bound",value=avg_monthly_lower_bound,min_value=0.00,max_value=1000.00,step=0.1) 
            cv_uper_bound = st.number_input(label=f"CV upper bound",value=cv_uper_bound,min_value=0.00,max_value=1000.00,step=1.0) 
            cv_lower_bound = st.number_input(label=f"CV lower bound",value=cv_lower_bound,min_value=0.00,max_value=1000.00,step=1.0)
            
            # percentile_sales_vol_upper = st.number_input(label=f"Percentile of sales volume upper bound (%)",value=percentile_sales_vol_upper,min_value=0,max_value=100,step=1)
            # percentile_sales_vol_lower = st.number_input(label=f"Percentile of sales volume lower bound (%)",value=percentile_sales_vol_lower,min_value=0,max_value=100,step=1)
            # percentile_cv_upper = st.number_input(label=f"Percentile of sales volume upper bound (%)",value=percentile_cv_upper,min_value=0,max_value=100,step=1)
            # percentile_cv_lower = st.number_input(label=f"Percentile of sales volume upper bound (%)",value=percentile_cv_lower,min_value=0,max_value=100,step=1)

            # submit_btn = st.form_submit_button("Change Decision")

            # wacc = st.multiselect(label="Weighted Average Cost Of Capital (%)",options=range(0,100,1))
            wacc = st.number_input(label="Weighted Average Cost Of Capital (%)",value = wacc ,step= 0.1)
            holding_cost = st.number_input(label="Holding cost (THB)",value = holding_cost ,step=1000.00)
            # # net_sales = st.number_input(label="Net sales (THB)",value = net_sales ,step=1000)
            lst_exclude_period_cogs = st.multiselect('Exclude Period COGS',lst_period_month)
            lst_exclude_period_rwds = st.multiselect('Exclude Period RW/DS',lst_period_month)
            
            save = st.checkbox('Save')

            if save:
                st.write('Great!')
            submit_btn = st.form_submit_button("Submit")

            i = 1

            for lst in lst_exclude_period_cogs:
                if i != 1 :
                    lst_str_cogs = lst_str_cogs + ",'" + lst + "'"
                else:
                    lst_str_cogs = "'" + lst + "'"
                i += 1

            i = 1

            for lst in lst_exclude_period_rwds:
                if i != 1 :
                    lst_str_rwds = lst_str_rwds + ",'" + lst + "'"
                else:
                    lst_str_rwds = "'" + lst + "'"
                i += 1

            df_lt = clustering_obj.set_lt(year_leadtime,z_score)

            # if lst_str_cogs is not None:รื
            #     cogs = get_cogs(lst_str_cogs)
            #     leadtime = get_leadtime(year_leadtime)

            #     cogs.Gram = cogs.Gram.astype(str)
            #     cogs['Gram'] = cogs['Gram'].str.strip()
            #     cogs['Grade'] = cogs['Grade'].str.strip()
            #     src_df = pd.read_pickle(os.path.join(path, './outbound/final_prepare.pkl'))
            #     src_df['Gram'] = src_df['Gram'].str.strip()
            #     src_df['Grade'] = src_df['Grade'].str.strip()

            #     # df_demand = 

            #     # =======================================================================================
            #     df_cal = src_df.merge(cogs,how='left',on=['Grade','Gram'])
            #     df_cal['grade_gram'] = df_cal['Grade'] + df_cal['Gram']
            #     # df_cal.to_excel('calulate_df_before.xlsx',index=False)
                
            #     leadtime['grade_gram'] = leadtime['grade_gram'].str.strip()
            #     df_cal['grade_gram'] = df_cal['grade_gram'].str.strip()

            #     df_cal = df_cal.merge(leadtime,how='left',on='grade_gram')
            #     df_cal.to_excel('calulate_df.xlsx',index=False)

                # df_clustering = df_cal.groupby(['sales_frequency']).agg(
                #     {
                #     'mat_number':'count',
                #     'average_weekly_sales_volume':'sum',
                #     'sd_weekly_sales_volume':'std'
                #     }
                # ).reset_index().rename(columns={'mat_number':'number_of_sku'}).fillna(0)

                # df_clustering['avg_sales'] = df_clustering['average_weekly_sales_volume'] / df_clustering['number_of_sku'] 

                # df.groupby(['revenue','session','user_id'])['user_id'].count()
                # df_clustering['cv'] = 
                # print(df_clustering)

            
            # if lst_str_rwds is not None:
            #     rwds = get_rwds(lst_str_rwds)
            #     rwds.to_excel('rwds.xlsx',index=False)

            #     rwds['Gram'] = rwds['Gram'].str.strip()
            #     rwds['Grade'] = rwds['Grade'].str.strip()
                
            #     df_cal = df_cal.merge(rwds,how='left',on=['Grade','Gram'])

            # src_df.to_excel('demand.xlsx',index=False)
        
            # df_ans = cal_(src_df,rwds,cogs)

        # with st.container():
        #     st.title("Analytic with clustering")

        #     lst_feature = [first_feature,second_feature]

        #     src_df = src_df.reset_index()

            
        #     # + alt.Chart(pd.DataFrame({'y': [120,20]})).mark_rule(color="blue").encode(y='y')+ alt.Chart(pd.DataFrame({'x': [50,110]})).mark_rule(color="blue").encode(x='x')

        #     st.altair_chart(altair_chart, use_container_width=True)

        pd.options.display.float_format = '{:.2f}'.format

        with st.container():
            st.title("Data for decision")
            st.dataframe(df_inf_clustering)
            st.metric(f'Percentile average monthly sales volume {percentile}' ,'{0:.2f}'.format(quantile_value) , delta=None, delta_color="normal")
            st.dataframe(src_df)

            src_sales_freq_df.to_pickle('./outbound/df_frequency.pkl')
            # src_sales_freq_df.to_excel('src_sales_freq_df.xlsx',index=False)

            src_sales_freq_df = src_sales_freq_df.rename(columns={"cv_weekly_x":"cv_weekly"})
            src_sales_freq_df = src_sales_freq_df.rename(columns={"cv_monthly_x":"cv_monthly"})
            src_sales_freq_df = src_sales_freq_df[~src_sales_freq_df['sales_frequency'].isnull()]

            fig = px.scatter(
                src_sales_freq_df, 
                x="avg_monthly", 
                y="cv_weekly", 
                color="sales_frequency",hover_data=['mat_number','Grade','Gram']
                )
            
            fig.add_hline(y=weekly_cv,line_dash="dash", line_color="red")
            fig.add_vline(x=quantile_value,line_dash="dash", line_color="red")

            st.plotly_chart(fig, use_container_width=True)

            altair_chart_elbow = alt.Chart(df_elbow).mark_point().encode(
                    x= 'Number of K', 
                    y= 'Distortions', 
                    tooltip=['Number of K','Distortions']
                ) 

            st.altair_chart(altair_chart_elbow, use_container_width=True)

            # print(return_clustring_df)

            fig_clustering = px.scatter(
                return_clustring_df, 
                x="avg_monthly", 
                y="cv_weekly", 
                color="Clustering",hover_data=['mat_number','Grade','Gram']
                )

            # return_clustring_df.to_excel('return_clustring_before_df.xlsx')

            condition_1_non_std = ((return_clustring_df["cv_weekly"] > cv_uper_bound) & (return_clustring_df["avg_monthly"] <  avg_monthly_lower_bound))
            condition_2_non_std = ((return_clustring_df["cv_weekly"] > cv_uper_bound) & (return_clustring_df["avg_monthly"] >=  avg_monthly_lower_bound)  & (return_clustring_df["avg_monthly"] <  avg_monthly_uper_bound))
            condition_3_low_std = ((return_clustring_df["cv_weekly"] > cv_uper_bound) & (return_clustring_df["avg_monthly"] >=  avg_monthly_uper_bound))
            condition_4_non_std = ((return_clustring_df["cv_weekly"] <= cv_uper_bound) & (return_clustring_df["cv_weekly"] > cv_lower_bound) & (return_clustring_df["avg_monthly"] <  avg_monthly_lower_bound))
            condition_5_low_std = ((return_clustring_df["cv_weekly"] <= cv_uper_bound) & (return_clustring_df["cv_weekly"] > cv_lower_bound) &  (return_clustring_df["avg_monthly"] >=  avg_monthly_lower_bound)  & (return_clustring_df["avg_monthly"] <  avg_monthly_uper_bound))
            condition_6_std = ((return_clustring_df["cv_weekly"] <= cv_uper_bound) & (return_clustring_df["cv_weekly"] > cv_lower_bound) &  (return_clustring_df["avg_monthly"] >=  avg_monthly_uper_bound))
            condition_7_low_std = ((return_clustring_df["cv_weekly"] <= cv_lower_bound) & (return_clustring_df["avg_monthly"] <  avg_monthly_lower_bound))
            condition_8_std = ((return_clustring_df["cv_weekly"] <= cv_lower_bound) & (return_clustring_df["avg_monthly"] >=  avg_monthly_lower_bound)  & (return_clustring_df["avg_monthly"] <  avg_monthly_uper_bound))
            condition_9_std = ((return_clustring_df["cv_weekly"] <= cv_lower_bound) & (return_clustring_df["avg_monthly"] >=  avg_monthly_uper_bound))

            return_clustring_df.loc[condition_1_non_std,'product_type'] = "NON-STD"
            return_clustring_df.loc[condition_2_non_std,'product_type'] = "NON-STD"
            return_clustring_df.loc[condition_3_low_std,'product_type'] = "LOW-STD"
            return_clustring_df.loc[condition_4_non_std,'product_type'] = "NON-STD"
            return_clustring_df.loc[condition_5_low_std,'product_type'] = "LOW-STD"
            return_clustring_df.loc[condition_6_std,'product_type'] = "STD"
            return_clustring_df.loc[condition_7_low_std,'product_type'] = "LOW-STD"
            return_clustring_df.loc[condition_8_std,'product_type'] = "STD"
            return_clustring_df.loc[condition_9_std,'product_type'] = "STD"

            return_clustring_df.loc[condition_1_non_std,'box'] = "1"
            return_clustring_df.loc[condition_2_non_std,'box'] = "2"
            return_clustring_df.loc[condition_3_low_std,'box'] = "3"
            return_clustring_df.loc[condition_4_non_std,'box'] = "4"
            return_clustring_df.loc[condition_5_low_std,'box'] = "5"
            return_clustring_df.loc[condition_6_std,'box'] = "6"
            return_clustring_df.loc[condition_7_low_std,'box'] = "7"
            return_clustring_df.loc[condition_8_std,'box'] = "8"
            return_clustring_df.loc[condition_9_std,'box'] = "9"

            return_clustring_df.to_pickle('./outbound/df_after_nine_box.pkl')
            
            # st.session_state.df_choose = st.session_state.df_choose.append(row_choose,ignore_index = True)
            # return_clustring_df.to_excel('return_clustring_df.xlsx')

            # fig_clustering.add_hline(y=0.4829,line_dash="dash", line_color="red")
            # fig_clustering.add_hline(y=0.8307,line_dash="dash", line_color="red")
            # fig_clustering.add_vline(x=79,line_dash="dash", line_color="blue")
            # fig_clustering.add_vline(x=40.5,line_dash="dash", line_color="blue")

            #  =====================================================================
            # return_clustring_df.info()
            # print(df_lt) 
            # print(df_lt)    
            #  =====================================================================
            fig_clustering.add_hline(y=cv_lower_bound,line_dash="dash", line_color="red")
            fig_clustering.add_hline(y=cv_uper_bound,line_dash="dash", line_color="red")
            fig_clustering.add_vline(x=avg_monthly_uper_bound,line_dash="dash", line_color="blue")
            fig_clustering.add_vline(x=avg_monthly_lower_bound,line_dash="dash", line_color="blue")

            # ===========================================================================================
            return_clustring_df['product_type'] = None

            st.plotly_chart(fig_clustering, use_container_width=True)

            # df_percentile = return_clustring_df.groupby('Clustering')['Clustering','cv_weekly'].quantile(.95)
            # df_percentile = return_clustring_df.groupby('Clustering')[['Clustering', 'cv_weekly']].quantile(.95)
            df_percentile = return_clustring_df.groupby('Clustering')['cv_weekly'].quantile(.95)

            print('#' * 100)
            print(df_percentile)
            print('#' * 100)

            row_choose,prod_lt = clustering_obj.conclusion(
                lst_exclude_period_rwds
                ,lst_exclude_period_cogs
                ,cogs
                ,cogs_wip_raw_mat
                ,coe_domestic
                ,service_level
                ,number_of_k
                # add this 
                ,avg_monthly_uper_bound
                ,avg_monthly_lower_bound
                ,cv_uper_bound
                ,cv_lower_bound
                ,wacc
                ,holding_cost
                )
            
            try:
                df_choose = pd.read_pickle('./outbound/df_choose.pkl')
                # print(1)
                # df_choose = df_choose.append(row_choose,ignore_index = True)
                df_choose = pd.concat([df_choose, pd.DataFrame([row_choose])], ignore_index=True)

                # df_choose = df_choose.drop_duplicates()
                df_choose.to_pickle('./outbound/df_choose.pkl')
            except:
                # df_choose = pd.read_pickle('./outbound/df_choose.pkl')
                print(row_choose)
                df_choose = pd.concat([df_choose, pd.DataFrame([row_choose])], ignore_index=True)
                df_choose.to_pickle('./outbound/df_choose.pkl')
                print(2)
            # x 100  ratio >> STD : Low STD : None 22/09/2022 >> Bow fix bug. Done
            # Add 6 type of sales frequency >> final_ans_df_cogs >> Done
            # final_ans_df_cogs if Y is null use T >> new ss
            
            st.dataframe(df_percentile.to_frame().style.format("{:.2f}"))
            st.dataframe(df_centroid)
            st.dataframe(percentile_df.style.format("{:.2f}"))
            st.dataframe(df_choose)
            st.write("Production Lead Time : %s" %prod_lt)
    