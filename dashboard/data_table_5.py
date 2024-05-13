# file table2.py
import altair as alt
import plotly.express as px

class Table5:
    def __init__(self, prepare_data) -> None:
        self.data = prepare_data
 
        
    def get_data(self):
        
        return_clustring_df = self.data.params_clustering_df
        cv_upper_bound = self.data.parameters_dict['cv_upper_bound']
        cv_lower_bound = self.data.parameters_dict['cv_lower_bound']
        avg_monthly_lower_bound = self.data.parameters_dict['avg_monthly_lower_bound']
        avg_monthly_upper_bound = self.data.parameters_dict['avg_monthly_upper_bound']
        
        fig_clustering = px.scatter(
                return_clustring_df, 
                x="avg_monthly", 
                y="cv_weekly", 
                color="Clustering",hover_data=['mat_number','Grade','Gram']
                ) 
 

        fig_clustering.add_hline(y=cv_lower_bound,line_dash="dash", line_color="red")
        fig_clustering.add_hline(y=cv_upper_bound,line_dash="dash", line_color="red")
        fig_clustering.add_vline(x=avg_monthly_upper_bound,line_dash="dash", line_color="blue")
        fig_clustering.add_vline(x=avg_monthly_lower_bound,line_dash="dash", line_color="blue")

        # ===========================================================================================
        # return_clustring_df['product_type'] = None
        
            
        return fig_clustering