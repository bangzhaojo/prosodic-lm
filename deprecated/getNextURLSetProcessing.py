import sqlalchemy 
from sqlalchemy import create_engine, text
import pandas as pd
import atexit

# Import dataframe into MySQL
database_username = 'blitt'
database_password = 'podcasts'
database_ip       = 'curry.si.umich.edu'
database_name     = 'podcasts'
table_name = 'mayJuneRemaining'

sqlEngine = sqlalchemy.create_engine('mysql+pymysql://blitt:password@curry.si.umich.edu:3306/podcasts', pool_recycle=3600, future=True)
with sqlEngine.connect() as conn:
    with conn.begin():
        
        def beforeExiting(): 
            #we get an exception here if we've managed to successfully 
            #close the connection before hitting this function
            #just need to prevent the exception from printing
            try:  
                conn.execute(text("UNLOCK TABLES;"))
                conn.close()
            except Exception as e: 
                pass    
        
        atexit.register(beforeExiting)
        
        conn.execute(text(f"LOCK TABLES podcasts.{table_name} WRITE;"))
        df = pd.read_sql(f"select * from podcasts.{table_name} WHERE processed = 0 LIMIT 1", conn, index_col="index")
        
        #url value 
        url = df.iloc[0, 0]
        print(url)
        
        #index value corresponding to url (this is a key)
        iVal = df.index.item() 
        
        #use key for faster search 
        #TODO: uncomment
        updateSQL = conn.execute(text(f"UPDATE podcasts.{table_name} SET processed = 1 WHERE `index` = {iVal};"))
        
        conn.execute(text("UNLOCK TABLES;"))
        conn.close()
