# file table2.py
class Table2:
    def __init__(self, prepare_data) -> None:
        self.data = prepare_data 
        
    def get_data(self): 
        return self.data.percent_df
 
    def get_quantile(self):
        return self.data.value_of_quntile


       
