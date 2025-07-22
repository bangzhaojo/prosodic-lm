from sqlalchemy import create_engine
import pymysql
import pandas as pd
import sqlalchemy

# df = pd.read_json("/shared/3/projects/benlitterer/podcastData/processed/mayJune/mayJuneDataClean.jsonl", orient="records", lines=True)

data = {
    "enclosure": [
        "http://example.com/podcast1.mp3",
        "http://example.com/podcast2.mp3",
        "http://example.com/podcast3.mp3",
        "http://example.com/podcast4.mp3",
        "http://example.com/podcast5.mp3"
    ],
    "processed": [0, 0, 0, 0, 0]  # Initial status is unprocessed
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

df.shape

df = df[["enclosure"]]
df["processed"] = 0

df = df.rename(columns={"enclosure":"url"}).reset_index(drop=True)


# Import dataframe into MySQL

# database_username = 'bangzhao'
# database_password = 'prosody'
# database_ip       = 'localhost'
# database_name     = 'prosody'

#establish connection to podcasts database
sqlEngine = sqlalchemy.create_engine('mysql+pymysql://bangzhao:prosody@localhost:3306/prosody', pool_recycle=3600)

conn = sqlEngine.connect()

#write the dataframe to mysql table 
df.to_sql(con=conn, name='mayJuneDataClean', if_exists='replace')

# Close the connection
conn.close()