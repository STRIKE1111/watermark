import sys
import os
import json
import csv
import yaml
import random
import xml.etree.ElementTree as ET
import sqlite3

# Optional imports for additional formats
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

# ensure project root is on sys.path so local modules import reliably
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_json(file):
    return [line.strip() for line in file]

def parse_csv(file):
    reader = csv.DictReader(file)
    return [row for row in reader]

def parse_tsv(file):
    reader = csv.DictReader(file, delimiter='\t')
    return [row for row in reader]

def parse_text(file):
    return [line.strip() for line in file]

def parse_xml(file):
    objects = []
    for line in file:
        line = line.strip()
        if line:
            try:
                xml_obj = ET.fromstring(line)
                objects.append(ET.tostring(xml_obj, encoding='unicode'))
            except ET.ParseError:
                pass
    return objects

def parse_yaml(file):
    try:
        yaml_data = yaml.safe_load(file)
        if isinstance(yaml_data, list):
            return yaml_data
        elif isinstance(yaml_data, dict):
            data = list(yaml_data.values())
            return [json.dumps(obj) for obj in data]
        else:
            raise ValueError("Invalid YAML format")
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file: {exc}")

def parse_custom_format(file):
    lines = file.readlines()
    header = lines[0].strip()
    return [header + '\n' + line.strip() for line in lines[1:]]

def parse_sqlite(filename):
    """Parse SQLite database and return rows as list of dictionaries."""
    conn = sqlite3.connect(filename)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get first table name
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()
    
    if not tables:
        conn.close()
        raise ValueError("No tables found in SQLite database")
    
    table_name = tables[0][0]
    
    # Query first table (limit to 1000 rows for sampling)
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1000")
    rows = cursor.fetchall()
    
    conn.close()
    
    # Convert rows to dictionaries and then to JSON strings
    return [json.dumps(dict(row)) for row in rows]


def parse_parquet(filename):
    """Parse Parquet file and return rows as list of dictionaries."""
    if not PARQUET_AVAILABLE:
        raise ImportError("pyarrow is required to read Parquet files. Install: pip install pyarrow")
    
    try:
        # Read parquet file
        table = pq.read_table(filename)
        
        # Convert to pandas dataframe for easier handling
        if PANDAS_AVAILABLE:
            df = table.to_pandas()
            # Limit to first 1000 rows for sampling
            df = df.head(1000)
            # Convert to list of JSON strings
            return [json.dumps(row.to_dict()) for _, row in df.iterrows()]
        else:
            # Fallback: convert to python dict directly
            data = table.to_pydict()
            num_rows = min(len(next(iter(data.values()))), 1000)
            
            # Convert columnar data to row-based
            rows = []
            for i in range(num_rows):
                row = {key: data[key][i] for key in data.keys()}
                rows.append(json.dumps(row))
            return rows
    except Exception as e:
        raise ValueError(f"Error parsing Parquet file: {e}")


def parse_bigquery_json(filename):
    """Parse BigQuery exported JSON file (newline-delimited JSON)."""
    if not PANDAS_AVAILABLE:
        # Fallback: read as regular JSON lines
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    try:
        # Read BigQuery JSON format (newline-delimited)
        df = pd.read_json(filename, lines=True)
        # Limit to first 1000 rows
        df = df.head(1000)
        # Convert to list of JSON strings
        return [json.dumps(row.to_dict()) for _, row in df.iterrows()]
    except Exception as e:
        raise ValueError(f"Error parsing BigQuery JSON file: {e}")

def read_file(filename):
    file_extension = filename[filename.rfind('.'):].lower()
    
    # Handle binary formats separately (don't open as text file)
    if file_extension in ['.db', '.sqlite', '.sqlite3']:
        return parse_sqlite(filename)
    elif file_extension == '.parquet':
        return parse_parquet(filename)
    elif file_extension in ['.jsonl', '.ndjson']:
        # BigQuery newline-delimited JSON format
        return parse_bigquery_json(filename)
    
    parsers = {
        '.json': parse_json,
        '.geojson': parse_json,  # GeoJSON 是 JSON 的一种变体
        '.jsonl': parse_json,    # JSON Lines 格式
        '.csv': parse_csv,
        '.tsv': parse_tsv,       # TSV 用 Tab 分隔符
        '.txt': parse_text,
        '.log': parse_text,
        '.xml': parse_xml,
        '.yaml': parse_yaml,
        '.yml': parse_yaml,      # YAML 的另一种扩展名
        '.abc': parse_custom_format
    }

    parser = parsers.get(file_extension)

    if not parser:
        raise ValueError(f"Unsupported file type: {file_extension}")

    with open(filename, 'r', encoding='utf-8') as file:
        return parser(file)

def load_and_sample_data(filename):
    objects = read_file(filename)
    if objects:
        return random.choice(objects)
    else:
        raise ValueError(f"No valid data found in the file: {filename}")

if __name__ == '__main__':
    filename = 'parser_data\\data\\self_made_data.abc'
    random_object = load_and_sample_data(filename)
    print("random object:\n", random_object)
