import pandas as pd
from lib.connection_db import set_connection


def get_demands():
    
    try:  
        engine = set_connection('AI_Demand') 
        demands_query = open("./sql_script/demands.sql").read()
        demands_df = pd.read_sql_query(demands_query, con=engine)
        demands_df['Grade'] = demands_df['Grade'].str.strip()
        demands_df.to_pickle('./outbound/demands.pkl') 
        print('== get demands success ==') 
    
    except Exception as e:
        print(e)
        return None
    
    
if __name__ == '__main__':
    
    get_demands()