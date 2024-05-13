import urllib
from sqlalchemy import create_engine
import configparser
import os 

def set_connection(env):
        
    try:
        config = configparser.ConfigParser()
        path = '/'.join((os.path.abspath(__file__).replace('\\', '/')).split('/')[:-1])
        config.read(os.path.join(path, './config.ini'))
        dest_server = config[env]['server'] 
        dest_database = config[env]['database']
        dest_username = config[env]['username']
        dest_password = config[env]['password']
        conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+dest_server+';DATABASE='+dest_database+';UID='+dest_username+';PWD='+ dest_password
        params = urllib.parse.quote_plus(conn_str)
        engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
        
        return engine
    
    except Exception as e:
        print(e)