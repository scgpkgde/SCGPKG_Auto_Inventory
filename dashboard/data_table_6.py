# file table1.py
class Table6:
    def __init__(self, prepare_data):
        self.data = prepare_data
 
    
    def get_data(self):
        return_clustring_df = self.data.params_clustering_df
        df_percentile = return_clustring_df.groupby('Clustering')['cv_weekly'].quantile(.95)
        return  df_percentile
    