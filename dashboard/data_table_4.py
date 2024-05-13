# file table2.py
import altair as alt


class Table4:
    def __init__(self, prepare_data) -> None:
        self.data = prepare_data
 
        
    def get_data(self):
        
        
        altair_chart_elbow = alt.Chart(self.data.df_elbow).mark_point().encode(
                    x= 'Number of K', 
                    y= 'Distortions', 
                    tooltip=['Number of K','Distortions']
                ) 
        return altair_chart_elbow



       
