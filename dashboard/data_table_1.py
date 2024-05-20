# file table1.py
class Table1:
    def __init__(self, prepare_data):
        self.data = prepare_data
 
    
    def get_data(self):
        
        final_df = self.data.final_df.rename(columns=
                                             {'sales_frequency': 'Sales Frequency',
                                              'total_ton': 'Total Ton',
                                              'avg_weekly': 'AVG Weekly',
                                              'std_weekly': 'STD Weekly',
                                              'avg_monthly': 'AVG Monthly',
                                              'std_monthly': 'STD Monthly',
                                              'cv_weekly': 'CV Weekly',
                                              'cv_monthly': 'CV Monthly' 
                                              })
        
        return  final_df
    
 