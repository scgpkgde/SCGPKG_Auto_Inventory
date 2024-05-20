# file table2.py
class Table2:
    def __init__(self, prepare_data) -> None:
        self.data = prepare_data 
        
    def get_data(self):
        percent_df = self.data.percent_df.rename(columns=
                                             {'product_type': 'Product Type',
                                              'total_ton': 'Total Ton',
                                              'percent_of_all': 'Percent of All' 
                                              })
         
        return percent_df
 
    def get_quantile(self):
        return self.data.value_of_quntile


       
