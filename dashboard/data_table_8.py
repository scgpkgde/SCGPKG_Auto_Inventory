# file table1.py
import pandas as pd

class Table8:
    def __init__(self, prepare_data):
        self.data = prepare_data
 
    
    def get_data(self): 
           
        row_choose = self.data.df_choose
 
        try:
                df_choose = pd.read_pickle('./outbound/df_choose.pkl') 
                df_choose = pd.concat([df_choose, row_choose], ignore_index=True) 
                df_choose.to_pickle('./outbound/df_choose.pkl')
        except:
                
                df_choose = pd.concat([df_choose, row_choose], ignore_index=True) 
                df_choose.to_pickle('./outbound/df_choose.pkl')

        df_choose = df_choose.rename(columns=
                                            { 
                                              'inventory_turnover': 'Inventory Turnover',
                                              'ratio (STD:LOW:NON)': 'Ratio (STD:LOW:NON)',
                                              'ratio (STD:LOW:NON) sales volumn': 'Ratio (STD:LOW:NON) Sales Volumn', 
                                              })
        
 
        return  df_choose
    