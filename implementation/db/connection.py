# author: Petra Oravová
# file: connection.py
# description: Utility functions for connecting to and interacting with a PostgreSQL database.
#              Includes functions to connect, close connections, execute queries,
#              and create the database and table if they don't exist.
# used documentation and tutorials:
# - https://www.psycopg.org/docs/
# - https://www.postgresql.org/docs/current/

import psycopg2


def get_connection():
    """
    Establishes a connection to the PostgreSQL database.

    Returns:
        psycopg2.connection: psycopg2 connection object.
    """
    conn = psycopg2.connect(
        dbname="cow_identification",
        user="postgres",
        password="951753456",
        host="localhost",
        port="5432"
    )
    return conn


def close_connection(conn):
    """
    Closes the connection to the PostgreSQL database.

    Args:
        conn (psycopg2.connection): psycopg2 connection object to be closed.
    """
    conn.close()


def execute_query(query, params=None, fetchone=False, fetchall=False):
    """
    Executes the given query on the PostgreSQL database.

    Args:
        query (str): SQL query to be executed.
        params (Optional[tuple], optional): Parameters to be used with the query. Defaults to None.
        fetchone (bool, optional): Whether to fetch one result. Defaults to False.
        fetchall (bool, optional): Whether to fetch all results. Defaults to False.

    Returns:
        Union[None, Tuple]: The result of the query, depending on fetchone or fetchall.
    """
    result = None

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params)

    if fetchone:
        result = cur.fetchone()
    elif fetchall:
        result = cur.fetchall()

    conn.commit()
    cur.close()
    conn.close()
    return result


def create_database():
    """
    Checks if the given database exists. If it doesn't, creates it.
    """
    try:
        # Connect to the default postgres database to check the existence of the target DB
        conn = psycopg2.connect(dbname="postgres", user='postgres', host='localhost', password="951753456")
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if the database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = 'cow_identification'")
        exists = cursor.fetchone() is not None

        if not exists:
            cursor.execute(f"CREATE DATABASE cow_identification")
            print(f"Database cow_identification created successfully.")
        else:
            print(f"Database cow_identification already exists.")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error checking or creating database: {e}")


def create_table():
    """
    Creates the 'cow_embeddings' table in the database if it doesn't already exist.
    Also creates the vector extension required for the embedding column.
    """
    query = '''CREATE EXTENSION IF NOT EXISTS vector;'''
    execute_query(query)
    query = '''
    CREATE TABLE IF NOT EXISTS cow_embeddings (
        id SERIAL PRIMARY KEY,
        cow_id TEXT NOT NULL,
        embedding VECTOR(256)
    );
    '''
    execute_query(query)
    print("The 'cow_embeddings' table has been successfully created.")
