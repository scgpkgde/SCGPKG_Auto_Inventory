# file data.py
import pandas as pd
from lib.connection_db import set_connection
from datetime import datetime, timedelta
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, quantile_transform
from sklearn import metrics
from sklearn.cluster import KMeans,MiniBatchKMeans
import os
 
    
class Data:
  
    def __init__(self, parameters_dict) -> None:

        self.parameters_dict = parameters_dict
        self.demands_df = self.get_demands()
        self.result_summary_df, self.final_prepare = self.get_result_summary()
        self.final_df, self.initial_data = self.get_inf_decision() 
        self.percent_df, self.value_of_quntile, self.percent_df_sales_freq, self.df_for_clustering, self.initial_std = self.set_std()
        self.params_clustering_df,self.df_elbow,self.cluster_centroid_df,self.df_percentile, self.df_after_nine_box = self.clustering()
        self.leadtime = self.get_leadtime()
        self.data_main, self.data_buffer_not_unknown = self.set_lt()
        self.rwds = self.get_rwds()
        self.cogs = self.get_cogs() 
        self.rwds_month_count = self.get_rwds_month_count()  
        self.adj_product_type  = self.get_adj_product_type()
        self.df_choose, self.production_lt = self.conclusion()
 
 
    def get_rwds(self):
        
        args_condition = self.parameters_dict['lst_exclude_period_rwds']
        engine = set_connection('COGS') 
        condition = ','.join("'" + str(x) + "'" for x in args_condition)
        rwds_query = open("./sql_script/sql_rwds.sql").read() 
        rwds_query = rwds_query %(condition,condition)
        rwds_df = pd.read_sql_query(rwds_query,con=engine)
        return rwds_df 

    def get_rwds_month_count(self):
        
        args_condition = self.parameters_dict['lst_exclude_period_rwds']
        engine = set_connection('COGS') 
        condition = ','.join("'" + str(x) + "'" for x in args_condition)
        rwds_query = open("./sql_script/sql_rwds_month_count.sql").read()  
        rwds_query = rwds_query %(condition,condition)
        rwds_df = pd.read_sql_query(rwds_query,con=engine)
        ans_month_cnt = rwds_df.iloc[0][0]
        return ans_month_cnt 
        
    def number_of_week(self):
        # Calculate the number of weeks
        diff_date = self.parameters_dict['end_period'] - self.parameters_dict['start_period']
        return round(diff_date.days / 7, 2)

    def number_of_month(self):
        # Calculate the number of months
        diff_date = self.parameters_dict['end_period'] - self.parameters_dict['start_period']
        return round(diff_date.days / 30, 0)    
 
    def get_demands(self):
          
        try:
            engine = set_connection('AI_Demand') 
            demands_query = open("./sql_script/demands.sql").read()
            str_start_period = self.parameters_dict['start_period'].strftime('%Y-%m-%d')
            str_end_period = self.parameters_dict['end_period'].strftime('%Y-%m-%d') 
            demands_query = demands_query %(str_start_period, str_start_period, str_start_period, str_end_period)
            demands_df = pd.read_sql_query(demands_query, con=engine)
            demands_df['Grade'] = demands_df['Grade'].str.strip()  
            return demands_df
        
        except Exception as e:
            print(e)
            return None

    def get_result_summary(self): 
        
        start_date = self.parameters_dict['start_period']
        end_date = self.parameters_dict['end_period']

        end_month = self.parameters_dict['start_period'].month
        end_year = self.parameters_dict['end_period'].year

        sales_frequency_df = self.demands_df.copy()
        
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
    
        # =================================================================================================
        monthly_df = sales_frequency_df[['Mat Number','Grade','Gram','month_number','ton']]
        monthly_df = monthly_df.groupby(['Mat Number','Grade','Gram','month_number']).agg(
            summary_ton_monthly=('ton','sum'),
        )

        monthly_df = monthly_df.groupby(['Mat Number','Grade','Gram']).agg(
            monthly_std=('summary_ton_monthly','std'),
            monthly_avg=('summary_ton_monthly','mean')
        )
    
        # =================================================================================================

        # print(sales_frequency_df)
        monday1 = (start_date - timedelta(days=start_date.weekday()))
        monday2 = (end_date - timedelta(days=end_date.weekday()))

        number_of_month = ((end_date.year - start_date.year) * 12 + end_date.month - start_date.month) + 1
        number_of_week = round((monday2 - monday1).days / 7,2)
    
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
        # print(f'Maximum no sales continuous of {previous_mat} : {max_continuous_non_sales}')
        prepare_df.loc[(prepare_df['Mat Number'] == previous_mat), 'max_non_sales'] = max_continuous_non_sales

        select_column = ['Mat Number','max_non_sales']
        prepare_df = prepare_df[select_column]
        prepare_df = prepare_df.drop_duplicates()
    

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
    
        result_summary_df = result_summary_df.drop_duplicates()
        result_summary_detail_df = result_summary_df.copy()
    

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
        # print("result_summary_detail_df")
    
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

        return result_summary_df, result_summary_detail_df
 
        
        # result_summary_df.to_pickle(os.path.join(path, './outbound/result_summary_df.pkl'))
        # result_summary_detail_df.to_pickle(os.path.join(path, './outbound/final_prepare.pkl'))
        
    def get_inf_decision(self):
        
        sales_frequent_df = self.final_prepare 
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
        final_df = self.result_summary_df 
        final_df = self.demands_df.merge(sales_frequent_df,how='left',on=['mat_number'])
        final_df = final_df.drop_duplicates()
        number_of_days = self.demands_df['dp_date'].nunique()

        # print(number_of_days)

 
        final_df['Grade_x'] = final_df['Grade_x'].str.strip()
        final_df['Gram_x'] = final_df['Gram_x'].str.strip()

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

        # print('cal_df_monthly')
        # print(cal_df_monthly)

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

        # print("conclusion_monthly_df_find_std")
        # print(conclusion_monthly_df)

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
        conclusion_weekly_ini['avg_weekly'] = conclusion_weekly_ini['sum_ton'] / self.number_of_week()
        conclusion_monthly_ini['avg_monthly'] = conclusion_monthly_ini['sum_ton'] / self.number_of_month()

        # print("conclusion_weekly_df")
        # print(conclusion_weekly_df)

        # print("conclusion_monthly_ini")
        # print(conclusion_monthly_ini)

        final_df = conclusion_weekly_df.merge(conclusion_monthly_df,how='inner',on='sales_frequency')   


        final_df_ini = conclusion_weekly_ini.merge(conclusion_monthly_ini,how='inner',on='mat_number')

        final_df_ini = final_df_ini.merge(conclusion_daily_ini,how='inner',on='mat_number') 


        final_df_ini['cv_weekly'] = final_df_ini['std_weekly'] / final_df_ini['avg_weekly']
        final_df_ini['cv_monthly'] = final_df_ini['std_monthly'] / final_df_ini['avg_monthly']
        final_df['cv_weekly'] = final_df['std_weekly'] / final_df['avg_weekly']
        final_df['cv_monthly'] = final_df['std_monthly'] / final_df['avg_monthly']
        
       

        final_df_ini.info()
  
        final_df_ini = sales_frequent_df[['mat_number','Grade','Gram','total_ton','avg_weekly','std_weekly','avg_monthly','std_monthly','cv_weekly','cv_monthly','average_daily','std_daily']]
        final_df_ini = final_df_ini.rename(columns={"Grade_x_x": "Grade", "Gram_x_x": "Gram"})

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
        
 
        # final_df_ini.to_pickle('./outbound/initial_data.pkl')
        # print(final_df)
        # print("final_df")
        
        
        return final_df, final_df_ini   
    
    def set_std(self):
        
        data = self.initial_data
        sales_frequent_df = self.final_prepare
  
        value_of_quntile = data['avg_monthly'].quantile(self.parameters_dict['quantile'])
        data["product_type"] = ""
    
        condition_std = (data["avg_monthly"] >= value_of_quntile) & (data["cv_weekly"] <= self.parameters_dict['weekly_cv'])
        condition_non_std = (data["avg_monthly"] < value_of_quntile) & (data["cv_weekly"] > self.parameters_dict['weekly_cv'])
        condition_unknow = (data["avg_monthly"] < value_of_quntile) & (data["cv_weekly"] <= self.parameters_dict['weekly_cv'])
        condition_outlier = (data["avg_monthly"] > value_of_quntile) & (data["cv_weekly"] > self.parameters_dict['weekly_cv'])

        data.loc[condition_std,'product_type'] = "STD"
        data.loc[condition_non_std,'product_type']= "NON-STD"
        data.loc[condition_unknow,'product_type'] = "Unknown"
        data.loc[condition_outlier,'product_type'] = "Outlier"

        percent_df_sales_freq = data.merge(sales_frequent_df,how='left',on='mat_number')
        percent_df_sales_freq = percent_df_sales_freq.rename(columns={"Grade_x": "Grade", "Gram_x": "Gram"})

        # percent_df_sales_freq.to_excel('percent_df_sales_freq',index=False)

        # data.to_pickle('./outbound/initial_std.pkl')
       
        total_ton = data['total_ton'].sum()
        percent_df = (data.groupby(['product_type'])
        .agg(
            total_ton=('total_ton','sum'),
            )
        .reset_index()
        )

        percent_df['percent_of_all'] = (percent_df['total_ton'] / total_ton) * 100

        df_for_clustering = percent_df_sales_freq.loc[percent_df_sales_freq["product_type"] =='Unknown']
                
   
        return percent_df, value_of_quntile, percent_df_sales_freq, df_for_clustering, data    

    def clustering(self):
        
        k = self.parameters_dict['number_of_k']
        params_clustering_df = self.df_for_clustering
        df_clustering = params_clustering_df.drop(columns =['mat_number','Grade','Gram'])
        df_clustering = df_clustering.fillna(0)
        lst_feature = ['avg_weekly','cv_weekly']

        df_clustering = df_clustering.rename(columns={"cv_weekly_x":"cv_weekly"})
        df_clustering = df_clustering.rename(columns={"cv_monthly_x":"cv_monthly"})
    

        df_clustering = df_clustering[lst_feature].reset_index()
    

        X = StandardScaler().fit_transform(df_clustering)
    
        k_means = KMeans(n_clusters=k,random_state=0, n_init=3)
        # k_means = KMeans(n_clusters=k)
        model = k_means.fit(X)
        # y_hat = k_means.predict(X)
        labels = k_means.labels_
        centroid = k_means.cluster_centers_ 
        silhouette_score = metrics.silhouette_score(X, labels, metric = 'euclidean')
        calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels)
    
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
    
            k_means_elbow = MiniBatchKMeans(n_clusters=k_elbow,random_state=0,batch_size=6, n_init=3)
            k_means_elbow.fit(X)
            
            distortions.append(sum(np.min(cdist(X, k_means_elbow.cluster_centers_,'euclidean'), axis=1)) / X.shape[0])
            inertias.append(k_means_elbow.inertia_)
                
            mapping1[k_elbow] = sum(np.min(cdist(X, k_means_elbow.cluster_centers_,'euclidean'), axis=1)) / X.shape[0]
            mapping2[k_elbow] = k_means_elbow.inertia_
    

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

        df_percentile = pd.DataFrame(columns=['percentile','sales_volume','cv'])

        lst_percentile =  range(10,100,5)

        for percentile in lst_percentile:
            percentile_val = percentile / 100

            avg_weekly_quntile = params_clustering_df['avg_weekly'].quantile(percentile_val)
            cv_weekly_quntile = params_clustering_df['cv_weekly'].quantile(percentile_val)

            data = {'percentile': percentile,
                    'sales_volume': avg_weekly_quntile,
                    'cv': cv_weekly_quntile}
    
            new_row = pd.DataFrame([data])
            df_percentile = pd.concat([df_percentile, new_row], ignore_index=True)
            
            
            
        return_clustring_df = params_clustering_df    
        cv_upper_bound = self.parameters_dict['cv_upper_bound']
        cv_lower_bound = self.parameters_dict['cv_lower_bound']
        avg_monthly_lower_bound = self.parameters_dict['avg_monthly_lower_bound']
        avg_monthly_upper_bound = self.parameters_dict['avg_monthly_upper_bound']
        
        condition_1_non_std = ((return_clustring_df["cv_weekly"] > cv_upper_bound) & (return_clustring_df["avg_monthly"] <  avg_monthly_lower_bound))
        condition_2_non_std = ((return_clustring_df["cv_weekly"] > cv_upper_bound) & (return_clustring_df["avg_monthly"] >=  avg_monthly_lower_bound)  & (return_clustring_df["avg_monthly"] <  avg_monthly_upper_bound))
        condition_3_low_std = ((return_clustring_df["cv_weekly"] > cv_upper_bound) & (return_clustring_df["avg_monthly"] >=  avg_monthly_upper_bound))
        condition_4_non_std = ((return_clustring_df["cv_weekly"] <= cv_upper_bound) & (return_clustring_df["cv_weekly"] > cv_lower_bound) & (return_clustring_df["avg_monthly"] <  avg_monthly_lower_bound))
        condition_5_low_std = ((return_clustring_df["cv_weekly"] <= cv_upper_bound) & (return_clustring_df["cv_weekly"] > cv_lower_bound) &  (return_clustring_df["avg_monthly"] >=  avg_monthly_lower_bound)  & (return_clustring_df["avg_monthly"] <  avg_monthly_upper_bound))
        condition_6_std = ((return_clustring_df["cv_weekly"] <= cv_upper_bound) & (return_clustring_df["cv_weekly"] > cv_lower_bound) &  (return_clustring_df["avg_monthly"] >=  avg_monthly_upper_bound))
        condition_7_low_std = ((return_clustring_df["cv_weekly"] <= cv_lower_bound) & (return_clustring_df["avg_monthly"] <  avg_monthly_lower_bound))
        condition_8_std = ((return_clustring_df["cv_weekly"] <= cv_lower_bound) & (return_clustring_df["avg_monthly"] >=  avg_monthly_lower_bound)  & (return_clustring_df["avg_monthly"] <  avg_monthly_upper_bound))
        condition_9_std = ((return_clustring_df["cv_weekly"] <= cv_lower_bound) & (return_clustring_df["avg_monthly"] >=  avg_monthly_upper_bound))

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
        df_after_nine_box = return_clustring_df    
            
        return params_clustering_df,df_elbow,cluster_centroid_df,df_percentile,df_after_nine_box

    def get_leadtime(self):
        
        engine = set_connection('COGS') 
        leadtime_query = open("./sql_script/production_leadtime.sql").read() 
        leadtime_query = leadtime_query %(self.parameters_dict['year_leadtime'])
        leadtime_df = pd.read_sql_query(leadtime_query,con=engine)
        return leadtime_df
  
    def set_lt(self): 
       
        params_z_score = self.parameters_dict['z_score']
        df_initial = self.initial_std
        df_lt = self.leadtime

        # df_initial.to_excel('df_initial.xlsx')
        # caculate : Q/2
        df_initial['Grade'] = df_initial['Grade'].str.strip()
        df_initial['Gram'] = df_initial['Gram'].str.strip()
        df_initial['grade_gram'] = df_initial['Grade'] + df_initial['Gram']
 
        df_result_lt = df_initial.merge(df_lt,how='left',on='grade_gram')
 
        df_result_lt['Q_2'] = (df_result_lt['average_daily'] * df_result_lt['avg_lt']) / 2
        df_result_lt['Safety_Stock'] = params_z_score * (
            (
                (df_result_lt['avg_lt'] * (df_result_lt['std_daily'] ** 2)) + 
                ((df_result_lt['average_daily'] ** 2) * (df_result_lt['sd_lt'] ** 2))
             ) ** (1/2) ) 
        
        data_main = df_result_lt.copy()  
        # df_result_lt.to_pickle('./outbound/data_main.pkl')
        condition_none_std = (df_result_lt["product_type"] == "")
        condition_not_unknown  = (df_result_lt["product_type"] != "Unknown")
        df_result_lt.loc[condition_none_std,'product_type'] = "NON-STD" 
        # df_result_lt.loc[condition_not_unknown].to_pickle('./outbound/data_buffer_not_unknown.pkl')
        data_buffer_not_unknown = df_result_lt.loc[condition_not_unknown]
        return data_main, data_buffer_not_unknown
    
    def get_cogs(self):
         
        args_condition = self.parameters_dict['lst_exclude_period_cogs']
        engine = set_connection('COGS') 
        cogs_query = open("./sql_script/cogs.sql").read() 
        condition = ','.join("'" + str(x) + "'" for x in args_condition) 
        cogs_query = cogs_query %(condition,condition)

        cogs_df = pd.read_sql_query(cogs_query,con=engine)
        return cogs_df

    def get_adj_product_type(self):
        engine = set_connection('COGS') 
        adj_query = open("./sql_script/adjust_product_size.sql").read()   
        adj_df = pd.read_sql_query(adj_query,con=engine)
        return adj_df
 
    def conclusion(self):
 
        params_cogs = self.parameters_dict['cogs']
        params_cogs_wip_raw_mat = self.parameters_dict['cogs_wip_raw_mat']
        params_coe_domestic = self.parameters_dict['coe_domestic'] 
        params_service_level = self.parameters_dict['service_level']
        params_k = self.parameters_dict['number_of_k']
        params_avg_monthly_upper_bound = self.parameters_dict['avg_monthly_upper_bound']
        params_avg_monthly_lower_bound = self.parameters_dict['avg_monthly_lower_bound']       
        params_cv_upper_bound = self.parameters_dict['cv_upper_bound']
        params_cv_lower_bound = self.parameters_dict['cv_lower_bound']
        params_wacc = self.parameters_dict['wacc']
        params_holding_cost = self.parameters_dict['holding_cost']

        df_main = self.data_main
        df_not_unknown =  self.data_buffer_not_unknown
        df_nine_box = self.df_after_nine_box

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

        df_rwds = self.rwds
        month_count = self.rwds_month_count

        ans_df['Gram'] = ans_df['Gram'].str.strip()
        ans_df['Grade'] = ans_df['Grade'].str.strip()
        
        ans_df_rwds = ans_df.merge(df_rwds,how='left',on=['Grade','Gram'])

        df_cogs = self.cogs

        ans_df_cogs = ans_df_rwds.merge(df_cogs,how='left',on=['Grade','Gram'])

        # ================================================================================

        adj_product_type_df = self.adj_product_type
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
        # final_ans_df_cogs.to_excel('final_ans_df_cogs_revise.xlsx',index=False)

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
        final_ans_df_cogs['avg_monthly_upper_bound'] = params_avg_monthly_upper_bound
        final_ans_df_cogs['avg_monthly_lower_bound'] = params_avg_monthly_lower_bound
        final_ans_df_cogs['cv_upper_bound'] = params_cv_upper_bound
        final_ans_df_cogs['cv_lower_bound'] = params_cv_lower_bound
        final_ans_df_cogs['wacc'] = params_wacc
        final_ans_df_cogs['holding_cost'] = params_holding_cost

        # df_frequency = pd.read_pickle('./outbound/df_frequency.pkl')
        df_frequency = self.percent_df_sales_freq
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
            ,'avg_monthly_upper_bound'
            ,'avg_monthly_lower_bound'
            ,'cv_upper_bound'
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

        # final_ans_df_cogs.to_excel('./debug/09-final_ans_df_cogs.xlsx', index=False, engine='openpyxl')
        # final_ans_df_cogs.to_pickle('final_ans_df_cogs.pkl')

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
        
        
        df_choose = pd.DataFrame([row])

        return df_choose, production_lt
    
    
    