import os
import psycopg2 as pg
import pandas as pd
from dotenv import load_dotenv


class SQLReader:
    def __init__(self, config) -> None:
        load_dotenv()
        self.config = config['global_params']['db']
        self.host = self.config['host']
        self.port = self.config['port']
        self.database = self.config['database']
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')

    def create_connection(self):
        connection = pg.connect(
            host=self.host, port=self.port, database=self.database, user=self.user, password=self.password
        )

        return connection

    def query(self, query):
        conn = self.create_connection()
        df = pd.read_sql(query, conn)

        return df
