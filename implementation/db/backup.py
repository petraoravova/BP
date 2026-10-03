# author: Petra Oravová
# file: backup.py
# description: Script to export and import PostgreSQL database using `pg_dump` and `pg_restore`.
# used documentation and tutorials:
# - https://docs.python.org/3/library/subprocess.html
# - https://www.postgresql.org/docs/current/app-pgdump.html
# - https://www.postgresql.org/docs/current/app-pgrestore.html

import os
import subprocess
from db.connection import create_table, create_database

# Set database password as environment variable
os.environ['PGPASSWORD'] = '951753456'


def export_database(output_filename):
    """
    Export the PostgreSQL database to a compressed backup file.

    Args:
        output_filename (str): The name of the output file to save the backup.
    """
    try:
        backup_path = "db/backups"
        os.makedirs(backup_path, exist_ok=True)
        output_path = os.path.join(backup_path, output_filename)

        command = [
            'pg_dump',
            '-U', "postgres",
            '-h', "localhost",
            '-d', "cow_identification",
            '-F', 'c',
            '-f', output_path
        ]
        subprocess.run(command, check=True)
        print(f"Database export to {output_path} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting database: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def import_database(input_filename):
    """
    Import the PostgreSQL database from a backup file.

    Args:
        input_filename (str): The name of the backup file to import.
    """
    try:
        create_database()
        create_table()
        backup_path = "db/backups"
        input_path = os.path.join(backup_path, input_filename)

        command = [
            'pg_restore',
            '-U', "postgres",
            '-h', "localhost",
            '-d', "cow_identification",
            '--clean',
            input_path
        ]
        subprocess.run(command, check=True)
        print(f"Database import from {input_path} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error importing database: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
