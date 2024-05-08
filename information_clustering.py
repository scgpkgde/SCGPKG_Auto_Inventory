# ========================================================================================
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from statistics import stdev
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans,KMeans,DBSCAN
from sklearn import metrics
from scipy.spatial.distance import cdist
import os
import configparser
import urllib
import numpy as np


class Clustering:

    def __init__(self):
        self.path = '/'.join((os.path.abspath(__file__).replace('\\', '/')).split('/')[:-1])

    # ========================================================================================
    def prepare_data(self,args_start_date,args_end_date):
        # Parameter
        # args_start_date = '2019-10-01'
        # args_end_date = '2021-09-30'

        # args_start_date = '2020-10-01'
        # args_end_date = '2022-09-30'

        config = configparser.ConfigParser()
        config.read(os.path.join(self.path, 'initial.ini'))
        dest_server = config['AI_Demand']['server'] 
        dest_database = config['AI_Demand']['database']
        dest_username = config['AI_Demand']['username']
        dest_password = config['AI_Demand']['password']
        conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+dest_server+';DATABASE='+dest_database+';UID='+dest_username+';PWD='+ dest_password
        params = urllib.parse.quote_plus(conn_str)
        engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
        demands_query = open(os.path.join(self.path, '.\Sql_scripts\demands.sql'), 'r').read()
        demands_query = demands_query %(args_start_date,args_start_date,args_start_date,args_end_date)
        # demands_query = demands_query %(args_start_date,args_end_date)
        # print(demands_query)
        demands_df = pd.read_sql_query(demands_query,con=engine)
        # demands_df.to_pickle(os.path.join(self.path, '.\\temp\demands_test.pkl'))

        start_date = datetime.strptime(args_start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args_end_date, '%Y-%m-%d')
        base_date = datetime.strptime(args_end_date, '%Y-%m-%d')

        # demands_df.to_excel("demand_df.xlsx",index=False)

        diff_date = end_date - start_date
        number_of_week = round(diff_date.days / 7,2)
        number_of_month = round(diff_date.days / 30,0)
        return demands_df,number_of_month,number_of_week

    # ================================================================================================================================

    def get_cogs(self,args_condition):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.path, 'initial.ini'))
        dest_server = config['COGS']['server'] 
        dest_database = config['COGS']['database']
        dest_username = config['COGS']['username']
        dest_password = config['COGS']['password']
        conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+dest_server+';DATABASE='+dest_database+';UID='+dest_username+';PWD='+ dest_password
        params = urllib.parse.quote_plus(conn_str)
        engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
        cogs_query = open(os.path.join(self.path, '.\Sql_scripts\cogs.sql'), 'r').read()

        condition = ','.join("'" + str(x) + "'" for x in args_condition)

        cogs_query = cogs_query %(condition,condition)

        cogs_df = pd.read_sql_query(cogs_query,con=engine)
        return cogs_df

    def get_rwds(self,args_condition):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.path, 'initial.ini'))
        dest_server = config['COGS']['server'] 
        dest_database = config['COGS']['database']
        dest_username = config['COGS']['username']
        dest_password = config['COGS']['password']

        condition = ','.join("'" + str(x) + "'" for x in args_condition)

        conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+dest_server+';DATABASE='+dest_database+';UID='+dest_username+';PWD='+ dest_password
        params = urllib.parse.quote_plus(conn_str)
        engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
        rwds_query = open(os.path.join(self.path, '.\Sql_scripts\sql_rwds.sql'), 'r').read()
        rwds_query = rwds_query %(condition,condition)
        rwds_df = pd.read_sql_query(rwds_query,con=engine)
        return rwds_df 

    def get_rwds_month_count(self,args_condition):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.path, 'initial.ini'))
        dest_server = config['COGS']['server'] 
        dest_database = config['COGS']['database']
        dest_username = config['COGS']['username']
        dest_password = config['COGS']['password']

        condition = ','.join("'" + str(x) + "'" for x in args_condition)

        conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+dest_server+';DATABASE='+dest_database+';UID='+dest_username+';PWD='+ dest_password
        params = urllib.parse.quote_plus(conn_str)
        engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
        rwds_query = open(os.path.join(self.path, '.\Sql_scripts\sql_rwds_month_count.sql'), 'r').read()
        rwds_query = rwds_query %(condition,condition)
        rwds_df = pd.read_sql_query(rwds_query,con=engine)
        ans_month_cnt = rwds_df.iloc[0][0]
        return ans_month_cnt 

    def get_leadtime(self,args_condition):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.path, 'initial.ini'))
        dest_server = config['COGS']['server'] 
        dest_database = config['COGS']['database']
        dest_username = config['COGS']['username']
        dest_password = config['COGS']['password']
        conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+dest_server+';DATABASE='+dest_database+';UID='+dest_username+';PWD='+ dest_password
        params = urllib.parse.quote_plus(conn_str)
        engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
        leadtime_query = open(os.path.join(self.path, '.\Sql_scripts\production_leadtime.sql'), 'r').read()
        leadtime_query = leadtime_query %(args_condition)
        leadtime_df = pd.read_sql_query(leadtime_query,con=engine)
        return leadtime_df

    def get_adj_product_type(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.path, 'initial.ini'))
        dest_server = config['COGS']['server'] 
        dest_database = config['COGS']['database']
        dest_username = config['COGS']['username']
        dest_password = config['COGS']['password']
        conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+dest_server+';DATABASE='+dest_database+';UID='+dest_username+';PWD='+ dest_password
        params = urllib.parse.quote_plus(conn_str)
        engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
        adj_query = open(os.path.join(self.path, r'.\Sql_scripts\adjust_product_size.sql'), 'r').read()
        adj_df = pd.read_sql_query(adj_query,con=engine)
        return adj_df

    def get_vc(self,args_condition):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.path, 'initial.ini'))
        dest_server = config['PPR']['server'] 
        dest_database = config['PPR']['database']
        dest_username = config['PPR']['username']
        dest_password = config['PPR']['password']
        conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+dest_server+';DATABASE='+dest_database+';UID='+dest_username+';PWD='+ dest_password
        params = urllib.parse.quote_plus(conn_str)
        engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
        vc_query = open(os.path.join(self.path, '.\Sql_scripts\cost.sql'), 'r').read()

        condition = ','.join("'" + str(x) + "'" for x in args_condition)

        vc_query = vc_query %(condition)

        vc_df = pd.read_sql_query(vc_query,con=engine)
        return vc_df

    # ============================================================================================

    def get_inf_decision(self,str_start_period,str_end_period):

        # Step 1 prepare demand
        demands_df_src,number_of_month,number_of_week = self.prepare_data(str_start_period,str_end_period)
        # demands_df_src.to_excel('demands_df_src.xlsx',index=False)
        sales_frequent_df = pd.read_pickle(os.path.join(self.path, './outbound/final_prepare.pkl'))
        sales_frequent_df = sales_frequent_df.rename(
            columns={
                'sd_weekly_sales_volume':'std_weekly',
                'average_weekly_sales_volume':'avg_weekly',
                'weekly_cv':'cv_weekly',
                'average_monthly_sales_volume':'avg_monthly' ,
                'monthly_cv':'cv_monthly',
                'sd_monthly_sales_volume':'std_monthly',
                'avg_daily':'average_daily'
                }
        )
        final_df = pd.read_pickle(os.path.join(self.path, './outbound/result_summary_df.pkl'))
        final_df = demands_df_src.merge(sales_frequent_df,how='left',on=['mat_number'])
        final_df = final_df.drop_duplicates()
        number_of_days = demands_df_src['dp_date'].nunique()

        # print(number_of_days)

        final_df.to_excel('final_df.xlsx',index=False)
        final_df['Grade_x'] = final_df['Grade_x'].str.strip()
        final_df['Gram_x'] = final_df['Gram_x'].str.strip()
        print(final_df)
        # Step 2 calculate
        cal_df_weekly = (final_df.groupby(['sales_frequency','number_of_week'])
        .agg(sum_ton=('ton','sum'),number_of_sku=('sales_frequency','count'))
        .reset_index()
        )
        

        cal_df_weekly_ini = (final_df.groupby(['mat_number','Grade_x','Gram_x','number_of_week'])
        .agg(sum_ton=('ton','sum'),number_of_sku=('sales_frequency','count'))
        .reset_index()
        )

        conclusion_weekly_ini = (cal_df_weekly_ini.groupby(['mat_number','Grade_x','Gram_x'])
        .agg(
            sum_ton =('sum_ton','sum'),
            avg_weekly=('sum_ton','mean'),
            std_weekly=('sum_ton','std'),
            )
        .reset_index()
        )

        # cal_df['number_of_week'] =  number_of_week
        # cal_df['number_of_month'] =  number_of_month
        # cal_df_weekly['avg_weekly']  = cal_df_weekly['sum_ton'] / number_of_week
        # cal_df['avg_monthly'] = cal_df['sum_ton'] / number_of_month
        # print(cal_df_weekly)

        conclusion_weekly_df = (cal_df_weekly.groupby(['sales_frequency'])
        .agg(
            total_ton =('sum_ton','sum'),
            avg_weekly=('sum_ton','mean'),
            std_weekly=('sum_ton','std'),
            )
        .reset_index()
        )
        
        # cal_df_monthly = (final_df.groupby(['sales_frequency','number_of_month_x'])
        cal_df_monthly = (final_df.groupby(['sales_frequency','number_of_month'])
        # cal_df_monthly = (final_df.groupby(['sales_frequency'])
        .agg(
            sum_ton=('ton','sum'),
            number_of_sku=('sales_frequency','count'))
        .reset_index()
        )

        print('cal_df_monthly')
        print(cal_df_monthly)

        # cal_df_monthly_ini = (final_df.groupby(['mat_number','Grade_x','Gram_x','number_of_month_x'])
 
        cal_df_monthly_ini = (final_df.groupby(['mat_number','Grade_x','Gram_x','number_of_month'])
        # cal_df_monthly_ini = (final_df.groupby(['mat_number','Grade_x','Gram_x'])
        .agg(
            total_ton=('ton','sum'),
            number_of_sku=('sales_frequency','count'))
        .reset_index()
        )

        conclusion_monthly_df = (cal_df_monthly.groupby(['sales_frequency'])
        .agg(
            avg_monthly=('sum_ton','mean'),
            std_monthly=('sum_ton','std'),
            )
        .reset_index()
        )

        print("conclusion_monthly_df_find_std")
        print(conclusion_monthly_df)

        conclusion_monthly_ini =(cal_df_monthly_ini.groupby(['mat_number','Grade_x','Gram_x'])
        .agg(
            sum_ton =('total_ton','sum'),
            avg_monthly=('total_ton','mean'),
            std_monthly=('total_ton','std'),
            )
        .reset_index()
        )

        conclusion_daily_ini = (final_df.groupby(['mat_number','Grade_x','Gram_x'])
        .agg(
            total_ton=('ton','sum'),
            std_daily=('ton','std')
        )
        .reset_index()
        )

        # print(conclusion_weekly_ini.info())
        conclusion_daily_ini["average_daily"] = conclusion_daily_ini['total_ton'] / number_of_days
        conclusion_weekly_ini['avg_weekly'] = conclusion_weekly_ini['sum_ton'] / number_of_week
        conclusion_monthly_ini['avg_monthly'] = conclusion_monthly_ini['sum_ton'] / number_of_month

        print("conclusion_weekly_df")
        print(conclusion_weekly_df)

        print("conclusion_monthly_ini")
        print(conclusion_monthly_ini)

        final_df = conclusion_weekly_df.merge(conclusion_monthly_df,how='inner',on='sales_frequency')   


        final_df_ini = conclusion_weekly_ini.merge(conclusion_monthly_ini,how='inner',on='mat_number')

        # final_df.to_excel('final_df.xlsx',index=False)
        final_df_ini = final_df_ini.merge(conclusion_daily_ini,how='inner',on='mat_number') 

        # conclusion_weekly_df.to_excel('conclusion_weekly_df.xlsx', index = False)
        # conclusion_monthly_df.to_excel('conclusion_monthly_df.xlsx', index = False)

        final_df_ini['cv_weekly'] = final_df_ini['std_weekly'] / final_df_ini['avg_weekly']
        final_df_ini['cv_monthly'] = final_df_ini['std_monthly'] / final_df_ini['avg_monthly']
        final_df['cv_weekly'] = final_df['std_weekly'] / final_df['avg_weekly']
        final_df['cv_monthly'] = final_df['std_monthly'] / final_df['avg_monthly']
        
       

        final_df_ini.info()
        final_df_ini.to_excel('final_df_ini.xlsx',index=False)
        final_df_ini = sales_frequent_df[['mat_number','Grade','Gram','total_ton','avg_weekly','std_weekly','avg_monthly','std_monthly','cv_weekly','cv_monthly','average_daily','std_daily']]
        final_df_ini = final_df_ini.rename(columns={"Grade_x_x": "Grade", "Gram_x_x": "Gram"})
        final_df_ini.to_excel('final_df_ini.xlsx',index=False)
        sales_frequent_df = sales_frequent_df.rename(
            columns={
                'sd_weekly_sales_volume':'std_weekly',
                'average_weekly_sales_volume':'avg_weekly',
                'weekly_cv':'cv_weekly',
                'average_monthly_sales_volume':'avg_monthly' ,
                'monthly_cv':'cv_monthly',
                'sd_monthly_sales_volume':'std_monthly'
                }
        )

        final_df_ini = sales_frequent_df[['mat_number','Grade','Gram','total_ton','avg_weekly','std_weekly','avg_monthly','std_monthly','cv_weekly','cv_monthly','average_daily','std_daily']]
        
        final_df_ini.to_excel('final_df_ini.xlsx',index=False)
        final_df.to_excel('final_df_end.xlsx',index=False)
        final_df_ini.to_pickle('./outbound/initial_data.pkl')
        print(final_df)
        print("final_df")
        return final_df

    def set_std(self,params_quntile,params_cv):
        data = pd.read_pickle('./outbound/initial_data.pkl')
        sales_frequent_df = pd.read_pickle(os.path.join(self.path, './outbound/final_prepare.pkl'))
        # print(data)
        value_of_quntile = data['avg_monthly'].quantile(params_quntile)
        data["product_type"] = ""

        # condition_std = (data["avg_monthly"] > value_of_quntile) & (data["cv_weekly"] < params_cv)
        # condition_non_std = (data["avg_monthly"] < value_of_quntile) & (data["cv_weekly"] > params_cv)
        # condition_unknow = (data["avg_monthly"] < value_of_quntile) & (data["cv_weekly"] < params_cv)
        # condition_outlier = (data["avg_monthly"] > value_of_quntile) & (data["cv_weekly"] > params_cv)
        
        condition_std = (data["avg_monthly"] >= value_of_quntile) & (data["cv_weekly"] <= params_cv)
        condition_non_std = (data["avg_monthly"] < value_of_quntile) & (data["cv_weekly"] > params_cv)
        condition_unknow = (data["avg_monthly"] < value_of_quntile) & (data["cv_weekly"] <= params_cv)
        condition_outlier = (data["avg_monthly"] > value_of_quntile) & (data["cv_weekly"] > params_cv)

        data.loc[condition_std,'product_type'] = "STD"
        data.loc[condition_non_std,'product_type']= "NON-STD"
        data.loc[condition_unknow,'product_type'] = "Unknown"
        data.loc[condition_outlier,'product_type'] = "Outlier"

        percent_df_sales_freq = data.merge(sales_frequent_df,how='left',on='mat_number')
        percent_df_sales_freq = percent_df_sales_freq.rename(columns={"Grade_x": "Grade", "Gram_x": "Gram"})

        # percent_df_sales_freq.to_excel('percent_df_sales_freq',index=False)

        data.to_pickle('./outbound/initial_std.pkl')
        
        total_ton = data['total_ton'].sum()
        percent_df = (data.groupby(['product_type'])
        .agg(
            total_ton=('total_ton','sum'),
            )
        .reset_index()
        )

        percent_df['percent_of_all'] = (percent_df['total_ton'] / total_ton) * 100

        df_for_clustering = percent_df_sales_freq.loc[percent_df_sales_freq["product_type"] =='Unknown']
        
        return percent_df,value_of_quntile,percent_df_sales_freq,df_for_clustering
    
    def clustering(self,params_clustering_df,params_number_of_k):
        k = params_number_of_k
        df_clustering = params_clustering_df.drop(columns =['mat_number','Grade','Gram'])
        df_clustering = df_clustering.fillna(0)
        lst_feature = ['avg_weekly','cv_weekly']

        df_clustering = df_clustering.rename(columns={"cv_weekly_x":"cv_weekly"})
        df_clustering = df_clustering.rename(columns={"cv_monthly_x":"cv_monthly"})

        # print(df_clustering.info())


        df_clustering = df_clustering[lst_feature].reset_index()
        # df_clustering.to_excel('df_clustering.xlsx')

        X = StandardScaler().fit_transform(df_clustering)
        # print(X)
        # X = df_clustering
        # k_means = MiniBatchKMeans(n_clusters=k,random_state=0)
        
        k_means = KMeans(n_clusters=k,random_state=0)
        # k_means = KMeans(n_clusters=k)
        model = k_means.fit(X)
        # y_hat = k_means.predict(X)
        labels = k_means.labels_
        centroid = k_means.cluster_centers_

        silhouette_score = metrics.silhouette_score(X, labels, metric = 'euclidean')
        calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels)

        # df_labels = pd.DataFrame(labels, columns = ['Clustering'])
    
        # params_clustering_df['Clustering'] =  df_labels['Clustering']

        list_string = map(str, labels)

        params_clustering_df['Clustering'] = np.array(list_string)
        # params_clustering_df.to_csv("params_clustering_df.csv")

        # elbow
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

        params_clustering_df = params_clustering_df.rename(columns={"cv_weekly_x":"cv_weekly"})
        params_clustering_df = params_clustering_df.rename(columns={"cv_monthly_x":"cv_monthly"})

        cluster_centroid_df = (params_clustering_df.groupby(['Clustering'])
        .agg(
            centroid_avg_monthly=('avg_monthly','mean'),
            avg_monthly_max=('avg_monthly','max'),
            avg_monthly_min=('avg_monthly','min'),
            centroid_cv_weekly=('cv_weekly','mean'),
            cv_weekly_max=('cv_weekly','max'),
            cv_weekly_min=('cv_weekly','min'),
            )
        .reset_index()
        )

        # value_of_quntile = data['avg_monthly'].quantile(params_quntile)

        df_percentile = pd.DataFrame(columns=['percentile','sales_volume','cv'])

        lst_percentile =  range(10,100,5)

        for percentile in lst_percentile:
            percentile_val = percentile / 100

            avg_weekly_quntile = params_clustering_df['avg_weekly'].quantile(percentile_val)
            cv_weekly_quntile = params_clustering_df['cv_weekly'].quantile(percentile_val)

            data = {'percentile': percentile,
                    'sales_volume': avg_weekly_quntile,
                    'cv': cv_weekly_quntile}

            # df_percentile = df_percentile.append(data, ignore_index=True)
            new_row = pd.DataFrame([data])
            df_percentile = pd.concat([df_percentile, new_row], ignore_index=True)

        # params_clustering_df.info()
        # accumurate
        # accum_df['avg_monthly'] = params_clustering_df['avg_monthly']
        # print(accum_df)
        # accum_df = accum_df.sort_values('avg_monthly',ascending=False,ignore_index=True)
        # accum_df['cum_sum'] = accum_df['avg_monthly'].cumsum()
        # accum_df['cum_perc'] = 100*accum_df['cum_sum']/accum_df['avg_monthly'].sum()
        # print(accum_df)

        return params_clustering_df,df_elbow,cluster_centroid_df,df_percentile

    def set_lt(self,params_lt_year,params_z_score):

        df_initial = pd.read_pickle('./outbound/initial_std.pkl')
        df_lt = self.get_leadtime(params_lt_year)

        # df_initial.to_excel('df_initial.xlsx')
        # caculate : Q/2
        df_initial['Grade'] = df_initial['Grade'].str.strip()
        df_initial['Gram'] = df_initial['Gram'].str.strip()
        df_initial['grade_gram'] = df_initial['Grade'] + df_initial['Gram']
        # df_result_lt = df_initial.merge(df_lt,how='inner',on='grade_gram')
        df_result_lt = df_initial.merge(df_lt,how='left',on='grade_gram')
        # df_lt.to_excel('df_lt.xlsx')
        # print(df_result_lt)
        # print(df_lt)
        df_result_lt['Q_2'] = (df_result_lt['average_daily'] * df_result_lt['avg_lt']) / 2
        df_result_lt['Safety_Stock'] = params_z_score * (
            (
                (df_result_lt['avg_lt'] * (df_result_lt['std_daily'] ** 2)) + 
                ((df_result_lt['average_daily'] ** 2) * (df_result_lt['sd_lt'] ** 2))
             ) ** (1/2) ) 
        

        df_result_lt.to_pickle('./outbound/data_main.pkl')
        condition_none_std = (df_result_lt["product_type"] == "")
        condition_not_unknown  = (df_result_lt["product_type"] != "Unknown")
        df_result_lt.loc[condition_none_std,'product_type'] = "NON-STD"

        df_result_lt.loc[condition_not_unknown].to_pickle('./outbound/data_buffer_not_unknown.pkl')

        return df_lt

    def conclusion(
        self
        ,params_lst_exclude_period_rwds
        ,params_lst_exclude_period_cogs
        ,params_cogs
        ,params_cogs_wip_raw_mat
        ,params_coe_domestic
        # add 
        ,params_service_level
        ,params_k
        ,params_avg_monthly_uper_bound
        ,params_avg_monthly_lower_bound
        ,params_cv_uper_bound
        ,params_cv_lower_bound
        ,params_wacc
        ,params_holding_cost
        ):

        df_main = pd.read_pickle('./outbound/data_main.pkl')
        df_not_unknown = pd.read_pickle('./outbound/data_buffer_not_unknown.pkl')
        df_nine_box = pd.read_pickle('./outbound/df_after_nine_box.pkl')

        # df_main.to_excel('09_df_main.xlsx',index=False)
        # df_not_unknown.to_excel('09_df_not_unknown.xlsx',index=False)
        # df_nine_box.to_excel('09_df_nine_box.xlsx',index=False)
        # df_not_unknown.to_excel('df_not_unknown.xlsx',index=False)
        # df_nine_box.to_excel('df_nine_box.xlsx',index=False)

        # print(df_main)

        df_not_unknown = df_not_unknown[['mat_number','Grade','Gram','product_type']]
        df_nine_box = df_nine_box[['mat_number','Grade','Gram','product_type','box']]

        union_df = pd.concat([df_not_unknown, df_nine_box],ignore_index=True)
        union_df.rename(columns = {'product_type':'product_type_new'}, inplace = True)

        ans_df = df_main.merge(union_df,how='left',on='mat_number')

        ans_df = ans_df[['mat_number','Grade_x','Gram_x','total_ton','avg_weekly','std_weekly','avg_monthly','std_monthly','cv_weekly','cv_monthly','average_daily','std_daily','grade_gram','ton','avg_lt','sd_lt','Q_2','Safety_Stock','product_type_new','box']]
        ans_df.rename(columns = {'product_type_new':'product_type','Grade_x':'Grade','Gram_x':'Gram'}, inplace = True)

        # ans_df.to_excel('ans_df.xlsx',index=False)
        # print(total_record)
        # print(std_count)
        # print(none_std_count)
        # print(ratio_std_count)
        # print(params_lst_exclude_period_rwds)

        df_rwds = self.get_rwds(params_lst_exclude_period_rwds)
        month_count = self.get_rwds_month_count(params_lst_exclude_period_rwds)

        ans_df['Gram'] = ans_df['Gram'].str.strip()
        ans_df['Grade'] = ans_df['Grade'].str.strip()
        
        ans_df_rwds = ans_df.merge(df_rwds,how='left',on=['Grade','Gram'])

        df_cogs = self.get_cogs(params_lst_exclude_period_cogs)

        ans_df_cogs = ans_df_rwds.merge(df_cogs,how='left',on=['Grade','Gram'])

        # ================================================================================

        adj_product_type_df = self.get_adj_product_type()
        ans_df_cogs = ans_df_cogs.merge(adj_product_type_df,how='left',on=['mat_number'])
        ans_df_cogs.rename(columns = {'product_type_y':'product_type_adj','product_type_x':'product_type'}, inplace = True)
        
        ans_df_cogs['product_type_new'] = ans_df_cogs.product_type.combine_first(ans_df_cogs.product_type_adj)
        # ans_df_cogs.to_excel('ans_df_cogs.xlsx',index = False)
        # ================================================================================
        condition_new_ss_non_std = (ans_df_cogs["product_type_new"] == 'NON-STD')
        condition_new_ss_std = (ans_df_cogs["product_type_new"] == 'STD')
        condition_new_ss_low_std = (ans_df_cogs["product_type_new"] == 'LOW-STD')

        ans_df_cogs_non_std = ans_df_cogs[condition_new_ss_non_std]
        ans_df_cogs_std = ans_df_cogs[condition_new_ss_std]
        ans_df_cogs_low_std = ans_df_cogs[condition_new_ss_low_std]
        # ================================================================================

        ans_df_cogs_non_std['new_ss'] = 0.00
        ans_df_cogs_std['new_ss'] = ans_df_cogs_std['Safety_Stock']
        ans_df_cogs_low_std['new_ss'] = ans_df_cogs_low_std['Safety_Stock'] * 0.5

        # ================================================================================
        # ans_df_cogs_non_std.to_excel('ans_df_cogs_non_std.xlsx',index=False)
        # ans_df_cogs_std.to_excel('ans_df_cogs_std.xlsx',index=False)
        # ans_df_cogs_low_std.to_excel('ans_df_cogs_low_std.xlsx',index=False)
        final_ans_df_cogs = pd.concat([ans_df_cogs_non_std, ans_df_cogs_std,ans_df_cogs_low_std],ignore_index=True)
        # final_ans_df_cogs.to_excel('final_ans_df_cogs_3.xlsx',index=False)
        final_ans_df_cogs["avg_inventory"] = final_ans_df_cogs["Q_2"] + final_ans_df_cogs["new_ss"]
        final_ans_df_cogs = final_ans_df_cogs.drop_duplicates()
        # final_ans_df_cogs.to_excel('final_ans_df_cogs_2.xlsx',index=False)

        # final_ans_df_cogs = final_ans_df_cogs.merge(adj_product_type_df,how='left',on=['mat_number'])

        # final_ans_df_cogs.to_excel('final_ans_df_cogs.xlsx',index=False)

        check_condition_same = (final_ans_df_cogs["product_type"] == final_ans_df_cogs["product_type_adj"])
        check_condition_same_non = ((final_ans_df_cogs["product_type"] == 'NON-STD') & (final_ans_df_cogs["product_type_adj"] == 'STD'))
        
        final_ans_df_cogs["rwds"] = final_ans_df_cogs["loss"] / (final_ans_df_cogs["total_ton"] * month_count)

        final_ans_df_cogs.loc[check_condition_same,'rwds'] = 0
        final_ans_df_cogs.loc[check_condition_same_non,'rwds']  = final_ans_df_cogs.loc[check_condition_same_non,'rwds'] * -1

        # ================================================================================
        final_ans_df_cogs = final_ans_df_cogs.drop_duplicates()
        final_ans_df_cogs.to_excel('final_ans_df_cogs_revise.xlsx',index=False)

        final_ans_df_cogs["avg_inventory_days"] = final_ans_df_cogs["avg_inventory"] / final_ans_df_cogs["average_daily"]
        final_ans_df_cogs["avg_x_cogs"] = final_ans_df_cogs["avg_inventory"] * final_ans_df_cogs["cogs_amt"]
        
        cogs_amt = final_ans_df_cogs["cogs_amt"].sum()
        q_2 = final_ans_df_cogs["Q_2"].sum()
        new_ss = final_ans_df_cogs["new_ss"].sum()
        cogs_amt = final_ans_df_cogs["cogs_amt"].sum()
        avg_inventory = final_ans_df_cogs["Q_2"].sum() + final_ans_df_cogs["new_ss"].sum()
        avg_inventory_days = avg_inventory / final_ans_df_cogs["average_daily"].sum()
        sum_avg_x_cogs = final_ans_df_cogs["avg_x_cogs"].sum()
        
        inventory_turnover_ratio = (params_cogs * (10**6)) / (sum_avg_x_cogs + ((params_cogs_wip_raw_mat * (10**6)) * params_coe_domestic))
        
        total_ton = final_ans_df_cogs["total_ton"].sum()
        lt_x_ton = (final_ans_df_cogs["total_ton"] * final_ans_df_cogs["avg_lt"]).sum()
        cogs_amt_avg = final_ans_df_cogs["cogs_amt"].mean()

        production_lt = lt_x_ton / total_ton

        # print(str_ratio)
        # print(q_2)
        # print(new_ss)
        # print(avg_inventory)
        # print(avg_inventory_days)
        # print(inventory_turnover_ratio)
        # print(total_ton)
        # print(lt_x_ton)

        # print("inventory_turnover_ratio : %s" %inventory_turnover_ratio)
        # print("production leadtime : %s" %production_lt)
        # print("cogs_amt_avg : %s" %cogs_amt_avg)
        # print("cogs_amt : %s" %cogs_amt)
        # print("total_ton : %s" %total_ton)
        # print("sum_avg_x_cogs : %s" %sum_avg_x_cogs)
        # final_ans_df_cogs["inventory_turnover_ratio"] = (params_cogs * (10**6)) / (final_ans_df_cogs["avg_inventory"] * final_ans_df_cogs["cogs_amt"] )

        # Revise no : STD => Non Std
        cnt_final_ans_df_cond = (((final_ans_df_cogs["product_type_adj"] == 'STD') | (final_ans_df_cogs["product_type_adj"] == 'LOW-STD')) & (final_ans_df_cogs["product_type"] == 'NON-STD'))
        cnt_final_ans_df_cogs = final_ans_df_cogs.loc[cnt_final_ans_df_cond]
        sku_revise_count = len(cnt_final_ans_df_cogs.index) 
        cnt_final_ans_df_cogs = cnt_final_ans_df_cogs[["grade_gram","total_ton"]]
        cnt_final_ans_df_cogs = cnt_final_ans_df_cogs.drop_duplicates()

        cnt_final_ans_df_cogs = (cnt_final_ans_df_cogs.groupby(['grade_gram'])
        .agg(
            sum_sales_ton=('total_ton','sum')
        )
        .sort_values(by='sum_sales_ton', ascending=False)
        .reset_index()
        )

        # ========================================================================================
        # df_vc = self.clustering_obj.get_vc(['2021_Yr'])
        # ========================================================================================

        total_ton = (cnt_final_ans_df_cogs['sum_sales_ton'].sum()) * 1
        cnt_final_ans_df_cogs['cum_sum'] = cnt_final_ans_df_cogs['sum_sales_ton'].cumsum()

        cnt_final_ans_df_cogs_condition = (cnt_final_ans_df_cogs['cum_sum'] < total_ton)

        cnt_final_ans_df_cogs = cnt_final_ans_df_cogs.loc[cnt_final_ans_df_cogs_condition]

        revise_no = cnt_final_ans_df_cogs['grade_gram'].count()
        # lst_grade_gram = ','.join(cnt_final_ans_df_cogs['grade_gram'].head(10).tolist())
        lst_grade_gram  = ','.join(cnt_final_ans_df_cogs['grade_gram'].tolist())
        revise_no = len(set(cnt_final_ans_df_cogs['grade_gram'].tolist()))

        # lst_grade_gram_all = ','.join(cnt_final_ans_df_cogs['grade_gram'].tolist())

        final_ans_df_cogs['K'] = params_k
        final_ans_df_cogs['service_level'] = params_service_level
        final_ans_df_cogs['avg_monthly_uper_bound'] = params_avg_monthly_uper_bound
        final_ans_df_cogs['avg_monthly_lower_bound'] = params_avg_monthly_lower_bound
        final_ans_df_cogs['cv_uper_bound'] = params_cv_uper_bound
        final_ans_df_cogs['cv_lower_bound'] = params_cv_lower_bound
        final_ans_df_cogs['wacc'] = params_wacc
        final_ans_df_cogs['holding_cost'] = params_holding_cost

        df_frequency = pd.read_pickle('./outbound/df_frequency.pkl')

        # final_ans_df_cogs.to_excel('final_ans_df_cogs_0.xlsx',index=False)
        
        # final_ans_df_cogs = final_ans_df_cogs.merge(df_frequency, on='mat_number', how='left')
        final_ans_df_cogs = final_ans_df_cogs.merge(df_frequency, on='mat_number', how='left', suffixes=('_left', '_right'))

        lst_column = [
            'mat_number'
            ,'total_ton'
            ,'avg_weekly_left'
            ,'std_weekly_left'
            ,'avg_monthly_left'
            ,'std_monthly_left'
            ,'cv_weekly'
            ,'cv_monthly'
            ,'average_daily_left'
            ,'std_daily'
            ,'grade_gram'
            ,'avg_lt'
            ,'sd_lt'
            ,'Q_2'
            ,'Safety_Stock'
            ,'product_type_left'
            ,'loss'
            ,'Grade_left'
            ,'Gram_left'
            ,'cogs_amt'
            ,'new_ss'
            ,'avg_inventory'
            ,'product_type_adj'
            ,'rwds'
            ,'avg_inventory_days'
            ,'avg_x_cogs'
            ,'K'
            ,'service_level'
            ,'avg_monthly_uper_bound'
            ,'avg_monthly_lower_bound'
            ,'cv_uper_bound'
            ,'cv_lower_bound'
            ,'wacc'
            ,'holding_cost'
            ,'sales_frequency'
            ,'box'
        ]

        final_ans_df_cogs = final_ans_df_cogs[lst_column]

         # final_ans_df_cogs = final_ans_df_cogs.rename(columns={"total_ton_x":"total_ton"})
        final_ans_df_cogs = final_ans_df_cogs.rename(columns={"avg_weekly_left":"avg_weekly"})
        final_ans_df_cogs = final_ans_df_cogs.rename(columns={"std_weekly_left":"std_weekly"})
        final_ans_df_cogs = final_ans_df_cogs.rename(columns={"avg_monthly_left":"avg_monthly"})
        final_ans_df_cogs = final_ans_df_cogs.rename(columns={"std_monthly_left":"std_monthly"})
        final_ans_df_cogs = final_ans_df_cogs.rename(columns={"cv_weekly_left":"cv_weekly"})
        final_ans_df_cogs = final_ans_df_cogs.rename(columns={"cv_monthly_left":"cv_monthly"})
        final_ans_df_cogs = final_ans_df_cogs.rename(columns={"average_daily_left":"average_daily"})
        final_ans_df_cogs = final_ans_df_cogs.rename(columns={"std_daily_left":"std_daily"})
        final_ans_df_cogs = final_ans_df_cogs.rename(columns={"product_type_left":"product_type"})
        final_ans_df_cogs = final_ans_df_cogs.rename(columns={"Grade_left":"Grade"})
        final_ans_df_cogs = final_ans_df_cogs.rename(columns={"Gram_left":"Gram"})

        final_ans_df_cogs = final_ans_df_cogs.drop_duplicates()

        final_ans_df_cogs["avg_weekly"] = final_ans_df_cogs["avg_weekly"].fillna(0)
        final_ans_df_cogs["std_weekly"] = final_ans_df_cogs["std_weekly"].fillna(0)
        final_ans_df_cogs["cv_weekly"] = final_ans_df_cogs["cv_weekly"].fillna(0)

        final_ans_df_cogs["avg_monthly"] = final_ans_df_cogs["avg_monthly"].fillna(0)
        final_ans_df_cogs["std_monthly"] = final_ans_df_cogs["std_monthly"].fillna(0)
        final_ans_df_cogs["cv_monthly"] = final_ans_df_cogs["cv_monthly"].fillna(0)

        final_ans_df_cogs.to_excel('./debug/09-final_ans_df_cogs.xlsx', index=False, engine='openpyxl')
        final_ans_df_cogs.to_pickle('final_ans_df_cogs.pkl')

        # =========================================================================
        condition_std = (final_ans_df_cogs['product_type'] == 'STD' )
        condition_non_std = (final_ans_df_cogs['product_type'] == 'NON-STD')
        condition_low_std = (final_ans_df_cogs['product_type'] == 'LOW-STD')

        total_record = final_ans_df_cogs['product_type'].count()
        std_count = final_ans_df_cogs.loc[condition_std]['product_type'].count()
        low_std_count = final_ans_df_cogs.loc[condition_low_std]['product_type'].count()
        none_std_count = final_ans_df_cogs.loc[condition_non_std]['product_type'].count()

        total_sales = final_ans_df_cogs['avg_monthly'].sum()
        std_sales = final_ans_df_cogs.loc[condition_std]['avg_monthly'].sum()
        low_std_sales = final_ans_df_cogs.loc[condition_low_std]['avg_monthly'].sum()
        none_std_sales = final_ans_df_cogs.loc[condition_non_std]['avg_monthly'].sum()

        avg_inventory = final_ans_df_cogs["Q_2"].sum() + final_ans_df_cogs["new_ss"].sum()
        avg_inventory_days = avg_inventory / final_ans_df_cogs["average_daily"].sum()

        ratio_std_sales = std_sales/total_sales
        ratio_none_std_sales = none_std_sales/total_sales
        ratio_low_std_sales = low_std_sales/total_sales
        str_ratio_sales = '{0:.2f}'.format(ratio_std_sales * 100) + ' : ' + '{0:.2f}'.format(ratio_low_std_sales * 100) + ' : '  + '{0:.2f}'.format(ratio_none_std_sales * 100)

        ratio_std_count = std_count / total_record
        ratio_none_std = none_std_count / total_record
        ratio_low_std = low_std_count / total_record
        str_ratio = '{0:.2f}'.format(ratio_std_count * 100) + ' : ' + '{0:.2f}'.format(ratio_low_std * 100) + ' : '  + '{0:.2f}'.format(ratio_none_std * 100)

        # =========================================================================

        row = { 
            'inventory_turnover': inventory_turnover_ratio,
	        'ratio (STD:LOW:NON)': str_ratio,
            'ratio (STD:LOW:NON) sales volumn': str_ratio_sales,
	        'inventory_days': avg_inventory_days,
            'avg_inventory' : avg_inventory,
	        'revise_no_of_std_non_std': revise_no,
            'revise_sku_no' : sku_revise_count,
            'main_grade': lst_grade_gram
        }

        return row, production_lt



        