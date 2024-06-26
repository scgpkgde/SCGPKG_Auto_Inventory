# file main.py 
# Reduced code
from dashboard import data_table_1, data_table_2, data_table_3, data_table_4, data_table_5, data_table_6, data_table_7, data_table_8, data_table_9
import dashboard.data_preparation as data_preparation
import streamlit as st
from datetime import datetime, date, timedelta
import scipy.stats as stat
import pandas as pd

from dashboard import data_table_6


def main(): 
    
    #*Default values
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
    cv_upper_bound = 0.8307
    avg_monthly_upper_bound = 79.0
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
    
    start_period = datetime.strptime('2021-10-01', '%Y-%m-%d')
    end_period = datetime.strptime('2022-09-30', '%Y-%m-%d')
 
    for period in lst_period:
        for month in lst_month:
            add_period = str(period) + '_' + str(month)
            lst_period_month.append(add_period)
            

    #=======================================================================================================
    #* Create sidebar
 
    with st.form("input_params_form"):
        with st.sidebar:
            #parameters decision
            st.title("Parameters for decision") 
            start_period = st.date_input("Data date from :",value=start_period)
            end_period = st.date_input("Data date to :",value=end_period)
 
            percentile = st.number_input(label=f"Percentile of sales volume (%)",value=percentile,min_value=0,max_value=100,step=1)
            quantile = percentile / 100
            weekly_cv = st.number_input(label="Weekly CV",value = weekly_cv ,step=0.1) 
            number_of_k = st.number_input(label="Choose number of cluster",value=number_of_k,step=1,format='%i')

            year_leadtime = st.selectbox('Leadtime Period ',lst_period,index= len(lst_period) - 2 )
            service_level = st.number_input(label=f"Percent Of Service Level (%)",value=service_level,min_value=0.00,max_value=100.00,step=1.00)
            z_score = stat.norm.ppf(service_level/100)
            cogs = st.number_input(label=f"Cost of goods sold FG (million THB)",value=cogs,min_value=0.00,max_value=10000000.00,step=1.00)
            cogs_wip_raw_mat = st.number_input(label=f"Cost of goods sold WIP & Raw material (million THB)",value=cogs_wip_raw_mat,min_value=0.00,max_value=10000000.00,step=1.00)
            coe_domestic = st.number_input(label=f"Domestic Portion",value=coe_domestic,min_value=0.00,max_value=1.00,step=0.01)

            #parameters clustering
            st.title("Parameters for clustering")
            avg_monthly_upper_bound = st.number_input(label=f"Sales volume upper bound",value=avg_monthly_upper_bound,min_value=0.00,max_value=1000.00,step=0.1) 
            avg_monthly_lower_bound = st.number_input(label=f"Sales volume lower bound",value=avg_monthly_lower_bound,min_value=0.00,max_value=1000.00,step=0.1) 
            cv_upper_bound = st.number_input(label=f"CV upper bound",value=cv_upper_bound,min_value=0.00,max_value=1000.00,step=1.0) 
            cv_lower_bound = st.number_input(label=f"CV lower bound",value=cv_lower_bound,min_value=0.00,max_value=1000.00,step=1.0)
        
            wacc = st.number_input(label="Weighted Average Cost Of Capital (%)",value = wacc ,step= 0.1)
            holding_cost = st.number_input(label="Holding cost (THB)",value = holding_cost ,step=1000.00)
            # # net_sales = st.number_input(label="Net sales (THB)",value = net_sales ,step=1000)
            lst_exclude_period_cogs = st.multiselect('Exclude Period COGS',lst_period_month, default=['2019_'])
            lst_exclude_period_rwds = st.multiselect('Exclude Period RW/DS',lst_period_month,default=['2019_'])
                        
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

            
            parameters_dict = {
                "quantile": quantile,
                "weekly_cv": weekly_cv,
                "number_of_k": number_of_k,
                "cv_lower_bound": cv_lower_bound,
                "cv_upper_bound": cv_upper_bound,
                "avg_monthly_upper_bound": avg_monthly_upper_bound,
                "avg_monthly_lower_bound": avg_monthly_lower_bound,
                "start_period": start_period,
                "end_period": end_period,
                "lst_exclude_period_rwds": lst_exclude_period_rwds,
                "lst_exclude_period_cogs": lst_exclude_period_cogs,
                "cogs": cogs,
                "cogs_wip_raw_mat": cogs_wip_raw_mat,
                "coe_domestic": coe_domestic,
                "service_level": service_level,
                "avg_monthly_upper_bound": avg_monthly_upper_bound,   
                "avg_monthly_lower_bound": avg_monthly_lower_bound,  
                "wacc": wacc,
                "holding_cost": holding_cost,
                "year_leadtime": year_leadtime,
                "z_score": z_score
            } 
    #=======================================================================================================
    submit_btn = True
    if submit_btn:
        with st.container():
            
            st.title("Data for decision") 
            prepare_data = data_preparation.Data(parameters_dict)   
            st.dataframe(data_table_1.Table1(prepare_data).get_data()) 
            st.dataframe(data_table_2.Table2(prepare_data).get_data()) 
            st.metric(f'Percentile average monthly sales volume {percentile}' ,'{0:.2f}'.format(data_table_2.Table2(prepare_data).get_quantile()) , delta=None, delta_color="normal")
            st.plotly_chart(data_table_3.Table3(prepare_data).get_data(), use_container_width=True)
            st.altair_chart(data_table_4.Table4(prepare_data).get_data(), use_container_width=True)
            st.plotly_chart(data_table_5.Table5(prepare_data).get_data(), use_container_width=True)
            st.dataframe(data_table_6.Table6(prepare_data).get_data().to_frame().style.format("{:.2f}"))
            st.dataframe(data_table_6.Table7(prepare_data).get_data())
            st.dataframe(data_table_7.Table8(prepare_data).get_data().style.format("{:.2f}"))
            st.dataframe(data_table_8.Table9(prepare_data).get_data())  
            st.write("Production Lead Time : %s" % data_table_9.Table10(prepare_data).get_data())
            
if __name__ == '__main__':  
    main()
