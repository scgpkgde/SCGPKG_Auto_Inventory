import streamlit as st 
import streamlit_shadcn_ui as ui
from dashboard import data_table_1, data_table_2, data_table_3, data_table_4, data_table_5, data_table_6, data_table_7, data_table_8, data_table_9
import dashboard.data_preparation as data_preparation
import streamlit as st
from datetime import datetime, date, timedelta
import scipy.stats as stat
import pandas as pd
from dashboard import data_table_6
import new_main
import os
from io import BytesIO

file_path = './outbound/final_ans_df_cogs.xlsx'

def write_date(start_date,end_date):
    data_string = start_date.strftime("%Y-%m-%d") + ',' + end_date.strftime("%Y-%m-%d")
    file_name = "./outbound/date_config.txt"
    with open(file_name, "w") as file:
        file.write(data_string)
    print(data_string)
    # st.write(f'--- write file -- {data_string}')

def read_date():
    file_name = "./outbound/date_config.txt"
    with open(file_name, "r") as file:
        content = file.read()
        content_ = content.split(", ")
    print(content_)
    return content_

def excel_file(file_path):
    df = pd.read_excel(file_path)                  
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
         df.to_excel(writer, index=False, sheet_name='Sheet1')
     
    output.seek(0)

    return output

def decision_cluster():
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
    lst_period = [i for i in range(current_year - 5,current_year)]
    lst_month = ['',1,2,3,4,5,6,7,8,9,10,11,12]
    lst_period_month = []
    exclude_year_defualt = str(current_year - 5)
    lst_exclude_period_cogs = []
    lst_exclude_period_rwds = []
    i = 1
    lst_str_cogs = None
    lst_str_rwds = None
    
    for period in lst_period:
        for month in lst_month:
            if month == '':
                add_period = str(period)
            else:
                add_period = str(period) + '_' + str(month)
            lst_period_month.append(add_period)
             
    with st.sidebar:    
        with st.form("input_params_form",border=False):
            
            if 'start_period' and 'end_period' not in st.session_state: 

                current_date = datetime.now()

                # Get the end of the current month
                current_year = current_date.year
                current_month = current_date.month-1

                # Calculate the last day of the current month
                if current_month == 12:
                    end_of_current_month = datetime(current_year, current_month, 31)
                else:
                    next_month = current_month + 1
                    end_of_next_month = datetime(current_year, next_month, 1)
                    end_of_current_month = end_of_next_month - timedelta(days=1)

                # Calculate the end of the same month 2 years ago
                two_years_ago = current_year - 2
                end_of_month_two_years_ago = datetime(two_years_ago, current_month, end_of_current_month.day)

                # Calculate the end of the same month this year
                end_of_month_this_year = datetime(current_year, current_month, end_of_current_month.day)

                st.session_state.start_period = end_of_month_two_years_ago.date()
                st.session_state.end_period = end_of_month_this_year.date()
                # date_ = {st.session_state.start_period,st.session_state.end_period}
                write_date(st.session_state.start_period,st.session_state.end_period)  
                  
            start_period = st.date_input("Data date from :", value=st.session_state.start_period)
            end_period = st.date_input("Data date to :", value=st.session_state.end_period)

              
     
            st.markdown('<hr style="margin-top: 5px; margin-bottom:15px;">', unsafe_allow_html=True)

            with st.expander("Decision Parameters"):
                st.header("Parameters for decision")
                percentile = st.number_input(label=f"Percentile of sales volume (%)",value=percentile,min_value=0,max_value=100,step=1)
                quantile = percentile / 100
                weekly_cv = st.number_input(label="Weekly CV",value = weekly_cv ,step=0.1) 
                number_of_k = st.number_input(label="Choose number of cluster",value=number_of_k,step=1,format='%i')

                # year_leadtime = st.selectbox('Leadtime Period ',lst_period,index= len(lst_period))
                year_leadtime = st.selectbox('Leadtime Period ',lst_period,index= len(lst_period) - 1)
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
                lst_exclude_period_cogs = st.multiselect('Exclude Period COGS',lst_period_month,default=[exclude_year_defualt])
                lst_exclude_period_rwds = st.multiselect('Exclude Period RW/DS',lst_period_month,default=[exclude_year_defualt])
            
            st.markdown('<hr style="margin-top: 5px; margin-bottom:15px;">', unsafe_allow_html=True)           
            submit_btn = st.form_submit_button("⭐ Submit ⭐",use_container_width=True,type="primary")

            i = 1
            for lst in lst_exclude_period_cogs:
     
                
                if i != 1 :
                    lst_str_cogs = lst_str_cogs + ",'" + lst + "'"
                else:
                    lst_str_cogs = "'" + lst + "'"
                i += 1

            i = 1
            for lst in lst_exclude_period_rwds:
                if '_' not in lst:
                    lst = str(lst)+'_'
                
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

    submit_btn = True
    if submit_btn :
        with st.container(): 
            date_ = read_date()
            date_string = date_[0]
            date_strings = date_string.split(',')

            # st.write(date_strings)

            start_period_config = datetime.strptime(date_strings[0], "%Y-%m-%d").date()
            end_period_config = datetime.strptime(date_strings[1], "%Y-%m-%d").date()

            # st.write('----- period ------')
            # st.write(start_period)
            # st.write(end_period)

            # st.write('----- config ------')
            # st.write(start_period_config)
            # st.write(end_period_config)

            is_change = False
            if start_period != start_period_config or end_period != end_period_config:
                # st.write('Chang')
                is_change = True
            # else:
            #     st.write('Not Chang')
    
            prepare_data = data_preparation.Data(parameters_dict, is_change)
 
            st.write(
                f'<h2 style="text-align: -webkit-center;color: #162e74;font-weight: 800;">Data for decision',
                f'</h2>',
                unsafe_allow_html=True
            )
              
            st.dataframe(data_table_1.Table1(prepare_data).get_data(), hide_index=True)  
            quantile_value = data_table_2.Table2(prepare_data).get_quantile()
            st.write(
                f'<div class="card" style="border: 1px solid rgb(0 0 0 / 19%);border-radius: 10px;margin-bottom: 20px;">',
                f'<div class="card-body" style="padding: 1%; text-align: -webkit-center;">',
                f'<div style="display: flex; justify-content: center;"><p class="card-text" style="margin: 0;">',
                f'"<b style="color: #5071d3; font-weight: 800;">{percentile}</b>" percentile of average monthly sales volume </p></div>',
                f'<p class="card-text" style="margin: 0; font-size: x-large; font-weight: 600;">{quantile_value:.2f} ton(s)</p>',
                '</div>',
                '</div>',
                unsafe_allow_html=True
            )
            st.dataframe(data=data_table_2.Table2(prepare_data).get_data(),width=1000, hide_index=True)
            st.plotly_chart(data_table_3.Table3(prepare_data).get_data(), use_container_width=True)
            st.altair_chart(data_table_4.Table4(prepare_data).get_data(), use_container_width=True)
            st.markdown('<hr style="margin-top: 5px; margin-bottom:15px;">', unsafe_allow_html=True)

            st.write(
                f'<h2 style="text-align: -webkit-center;color: #162e74;font-weight: 800;">Data for cluster',
                f'</h2>',
                unsafe_allow_html=True
            )
            st.plotly_chart(data_table_5.Table5(prepare_data).get_data(), use_container_width=True)
            st.dataframe(data_table_6.Table6(prepare_data).get_data(),width=1000, hide_index=True)
            header_style = """
                <style>
                    .dataframe th {
                        text-align: center;
                    }
                </style>
            """
            st.write(header_style, unsafe_allow_html=True)
            st.dataframe(data_table_7.Table7(prepare_data).get_data().style.format("{:.2f}"),width=1000, hide_index=True)
            st.dataframe(data_table_8.Table8(prepare_data).get_data(), hide_index=True)  
            st.write(
                f'<div class="card" style="border: 1px solid rgb(0 0 0 / 19%);border-radius: 10px;margin-bottom: 20px;">',
                f'<div class="card-body" style="padding: 1%; text-align: -webkit-center;">',
                f'<p class="card-text" style="margin: 0;">Production Lead Time</p>',
                f'<p class="card-text" style="margin: 0; font-size: x-large; font-weight: 600;">{data_table_9.Table9(prepare_data).get_data()}</p>',
                '</div>',
                '</div>',
                unsafe_allow_html=True
            )
            
            # parameters_dict["start_period"] = start_period_config
            # parameters_dict["end_period"] = end_period_config

            current_date = datetime.now()
            formatted_date = current_date.strftime('%Y%m%d')
            file_name = "Inventory_Detail_" + formatted_date + ".xlsx"

            try:
                output = excel_file(file_path)
                st.download_button(
                    label="Download Excel File",
                    data=output,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                    st.error(f"Error: {str(e)}")

            
            write_date(start_period,end_period)  
            submit_btn = False

def page1():
    if st.button('ไปที่หน้า 2'):
        st.session_state.page = 'page2'
        st.experimental_rerun()
    decision_cluster()
          
def newpage():
    if st.button('ไปที่หน้า 1'):
        st.session_state.page = 'page1'
        st.experimental_rerun()
    new_main.main()
    
def main():
    decision_cluster()
    # if 'page' not in st.session_state:
    #     st.session_state.page = 'page1'

    # if st.session_state.page == 'page1':
    #     page1()
    # elif st.session_state.page == 'page2':
    #     newpage()
 
main()