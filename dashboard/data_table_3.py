# file table2.py
import pandas as pd
import plotly.express as px
class Table3:
    def __init__(self, prepare_data) -> None:
        self.data = prepare_data
 
        
    def get_data(self):
          
        # src_sales_freq_df.rename(columns={'cv_weekly_x': 'cv_weekly', "cv_monthly_x":"cv_monthly"}, inplace=True)
        # src_sales_freq_df = self.data.percent_df_sales_freq[~src_sales_freq_df['sales_frequency'].isnull()]
 
        fig = px.scatter(
                self.data.percent_df_sales_freq, 
                x="avg_monthly", 
                y="cv_monthly_x", 
                color="sales_frequency",hover_data=['mat_number','Grade','Gram']
                )
        
        fig.add_hline(y=self.data.parameters_dict['weekly_cv'],line_dash="dash", line_color="red")
        fig.add_vline(x=self.data.value_of_quntile,line_dash="dash", line_color="red")

       
        return fig

