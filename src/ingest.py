
def data_ingest(con, query):
    con = sqlite3.connect('ProjectData/gas_monitoring.db')

    cursor = con.cursor()
    
            # Query the sqlite_master table to get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            
            # Fetch all results
    table_names = [row[0] for row in cursor.fetchall()]
    query = "SELECT * FROM gas_monitoring"  # Replace 'your_table_name' with the actual table name
    df = pd.read_sql_query(query, con)
    return df

