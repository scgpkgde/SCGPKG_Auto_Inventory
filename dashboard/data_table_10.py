# file table1.py
import pandas as pd

class Table10:
    def __init__(self, prepare_data):
        self.data = prepare_data
 
    
    def get_data(self): 
        
        return  round(self.data.production_lt,2)
    