import streamlit as st
import streamlit_shadcn_ui as ui
from dashboard import data_table_1, data_table_2, data_table_3, data_table_4, data_table_5, data_table_6, data_table_7, data_table_8, data_table_9, data_table_10
import dashboard.data_preparation as data_preparation
import streamlit as st
from datetime import datetime, date, timedelta
import scipy.stats as stat
import pandas as pd

def main2():

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
    number_of_k = 3
    service_level = 93.00
    quantile = percentile / 100
    current_year = date.today().year
    year_leadtime = current_year - 1
    cogs = 22100.00
    holding_cost = 160.0
    wacc = 11.5
    z_score = None
    lst_period = [i for i in range(current_year - 5,current_year + 1)]
    lst_month = ['',1,2,3,4,5,6,7,8,9,10,11,12]
    lst_period_month = []
    lst_exclude_period_cogs = ['2019_']
    lst_exclude_period_rwds = ['2019_']
    i = 1
    lst_str_cogs = None
    lst_str_rwds = None


    for period in lst_period:
        for month in lst_month:
            add_period = str(period) + '_' + str(month)
            lst_period_month.append(add_period)
            
    start_period = datetime.strptime('2021-10-01', '%Y-%m-%d')
    end_period = datetime.strptime('2022-09-30', '%Y-%m-%d')

#=======================================================================================================
    with st.form("input_params_form"):
        with st.sidebar:
            start_period = st.date_input("Data date from :",value=start_period)
            end_period = st.date_input("Data date to :",value=end_period)
            st.markdown('<hr style="margin-top: 5px; margin-bottom:15px;">', unsafe_allow_html=True)

            with st.expander("Decision Parameters"):
                st.header("Parameters for decision")
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
                
            with st.expander("Cluster Parameters"):
                st.header("Parameters for cluster")
                avg_monthly_upper_bound = st.number_input(label=f"Sales volume upper bound",value=avg_monthly_upper_bound,min_value=0.00,max_value=1000.00,step=0.1) 
                avg_monthly_lower_bound = st.number_input(label=f"Sales volume lower bound",value=avg_monthly_lower_bound,min_value=0.00,max_value=1000.00,step=0.1) 
                cv_upper_bound = st.number_input(label=f"CV upper bound",value=cv_upper_bound,min_value=0.00,max_value=1000.00,step=1.0) 
                cv_lower_bound = st.number_input(label=f"CV lower bound",value=cv_lower_bound,min_value=0.00,max_value=1000.00,step=1.0)
            
                wacc = st.number_input(label="Weighted Average Cost Of Capital (%)",value = wacc ,step= 0.1)
                holding_cost = st.number_input(label="Holding cost (THB)",value = holding_cost ,step=1000.00)
                lst_exclude_period_cogs = st.multiselect('Exclude Period COGS',lst_period_month, default=['2019_'])
                lst_exclude_period_rwds = st.multiselect('Exclude Period RW/DS',lst_period_month,default=['2019_'])
            
            st.markdown('<hr style="margin-top: 5px; margin-bottom:15px;">', unsafe_allow_html=True)           
            submit_btn = st.form_submit_button("⭐ Submit ⭐",use_container_width=True)

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
    if submit_btn :
        with st.container():
            prepare_data = data_preparation.Data(parameters_dict)   
            st.header("Data for decision")
            st.dataframe(data_table_1.Table1(prepare_data).get_data())
            quantile_value = data_table_2.Table2(prepare_data).get_quantile()
            st.write(
                f'<div class="card" style="border: 1px solid rgb(0 0 0 / 19%);border-radius: 10px;margin-bottom: 20px;">',
                f'<div class="card-body" style="padding: 1%; text-align: -webkit-center;">',
                f'<p class="card-text" style="margin: 0;">" {percentile} " percentile of average monthly sales volume </p>',
                f'<p class="card-text" style="margin: 0; font-size: x-large; font-weight: 600;">{quantile_value:.2f} ton(s)</p>',
                '</div>',
                '</div>',
                unsafe_allow_html=True
            )
            st.dataframe(data=data_table_2.Table2(prepare_data).get_data(),width=1000)
            st.plotly_chart(data_table_3.Table3(prepare_data).get_data(), use_container_width=True)
            st.altair_chart(data_table_4.Table4(prepare_data).get_data(), use_container_width=True)
            st.markdown('<hr style="margin-top: 5px; margin-bottom:15px;">', unsafe_allow_html=True)

            st.header("Data for cluster")
            st.plotly_chart(data_table_5.Table5(prepare_data).get_data(), use_container_width=True)

            cols = st.columns(2)
            with cols[0]:
                st.dataframe(data_table_6.Table6(prepare_data).get_data().to_frame().style.format("{:.2f}"),width=1000)
            with cols[1]:
                st.dataframe(data_table_7.Table7(prepare_data).get_data(),width=1000)

            st.dataframe(data_table_8.Table8(prepare_data).get_data().style.format("{:.2f}"),width=1000)
            st.dataframe(data_table_9.Table9(prepare_data).get_data())  
            st.write(
                f'<div class="card" style="border: 1px solid rgb(0 0 0 / 19%);border-radius: 10px;margin-bottom: 20px;">',
                f'<div class="card-body" style="padding: 1%; text-align: -webkit-center;">',
                f'<p class="card-text" style="margin: 0;">Production Lead Time</p>',
                f'<p class="card-text" style="margin: 0; font-size: x-large; font-weight: 600;">{data_table_10.Table10(prepare_data).get_data()}</p>',
                '</div>',
                '</div>',
                unsafe_allow_html=True
            )

            submit_btn = False

            if st.button("Go to new Page"):
                st.session_state.page = "main2"
        
    
if __name__ == '__main__':  
    # main()
    if 'page' not in st.session_state:
        st.session_state.page = 'main'
        
    if st.session_state.page == 'main':
        main2()
    elif st.session_state.page == 'main2':
        import new_main
        new_main.main()