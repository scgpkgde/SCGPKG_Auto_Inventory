# file table1.py
class Table6:
    def __init__(self, prepare_data):
        self.data = prepare_data
        
    
 
    
    def get_data(self):
        
        cluster_centroid_df = self.data.cluster_centroid_df.rename(columns=
                                             { 
                                              'cv_weekly': 'CV Weekly',
                                              'centroid_avg_monthly': 'Centroid AVG Monthly',
                                              'avg_monthly_max': 'AVG Monthly Max',
                                              'avg_monthly_min': 'AVG Monthly Min',
                                              'centroid_cv_weekly': 'Centroid CV Weekly',
                                              'cv_weekly_max': 'CV Weekly Max',
                                              'cv_weekly_min': 'CV Weekly Min', 
                                              })
        
        
        return  cluster_centroid_df
    