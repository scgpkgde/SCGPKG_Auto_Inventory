# file table1.py
class Table7:
    def __init__(self, prepare_data):
        self.data = prepare_data
 
    
    def get_data(self):
    
        df_percentile = self.data.df_percentile.rename(columns=
                                            { 
                                              'percentile': 'Percentile',
                                              'sales_volume': 'Sales Volume',
                                              'cv': 'CV' 
                                              })
        return  df_percentile
    