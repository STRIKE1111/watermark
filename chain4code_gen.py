import time
import logging
import os
import subprocess
import sys
import json
import argparse

# ensure project root on sys.path for local imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import CFG, frozen_seed, load_env_variables
from file_parsers import load_and_sample_data
from langchain_model import constrained_self_and_plan_chains
from openai_config import setup_openai

def run_generated_code(code_path, retries=0, max_retries=5):
    try:
        # use the current Python interpreter to run generated code to avoid mismatched interpreters
        result = subprocess.run([sys.executable, code_path], capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"Error running the script: {result.stderr}")
            return False
        else:
            logging.info(f"Code ran successfully: {result.stdout}")
            return True
    except Exception as e:
        logging.error(f"Exception occurred while running the script: {e}")
        return False


def process_dataset(csv_path, dataset_name, gpt_model, chain_function):
    """Process a single dataset and generate parsing code."""
    custom_chain = chain_function(model_name=gpt_model)
    
    chain_function_name = chain_function.__name__
    log_filename = f"{dataset_name}-{chain_function_name}-gpt.log"
    
    log_folder = 'Lllog'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Create a separate logger for this dataset
    dataset_logger = logging.getLogger(dataset_name)
    dataset_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    dataset_logger.handlers = []
    
    # Add file handler
    fh = logging.FileHandler(os.path.join(log_folder, log_filename), mode='w+', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    dataset_logger.addHandler(fh)

    try:
        data_piece = load_and_sample_data(csv_path)
        
        function_gen, code, result_json, separators, nest, tags = custom_chain.invoke({"data": data_piece})

        # Create file with dataset identifier in generated_code directory
        code_dir = "generated_code"
        if not os.path.exists(code_dir):
            os.makedirs(code_dir)
        code_path = os.path.join(code_dir, f"convert2json_{dataset_name}.py")
        with open(code_path, "w", encoding='utf-8') as file:
            file.write(code)
            dataset_logger.info(f"File '{code_path}' has been updated.")
        
        dataset_logger.info(f"random_object:\n {data_piece}")
        dataset_logger.info(f"separators:\n {separators}")
        dataset_logger.info(f"nest:\n {nest}")
        dataset_logger.info(f"tags:\n {tags}")
        dataset_logger.info(f"json_result:\n {result_json}")
        dataset_logger.info(f"function_gen:\n {function_gen}")
        dataset_logger.info(f"code:\n {code}")

        # Run the generated code and handle any errors, with a retry limit of 5 attempts
        success = run_generated_code(code_path, retries=0, max_retries=5)
        
        retries = 0
        while not success and retries < 5:
            dataset_logger.info(f"Attempt {retries + 1} failed. Regenerating code and trying again.")
            
            # Regenerate the code
            function_gen, code, result_json, separators, nest, tags = custom_chain.invoke({"data": data_piece})
            with open(code_path, "w", encoding='utf-8') as file:
                file.write(code)
                dataset_logger.info(f"File '{code_path}' has been updated.")

            # Run the new generated code
            success = run_generated_code(code_path, retries=retries + 1, max_retries=5)
            retries += 1

        # Log if all attempts failed
        if not success:
            dataset_logger.error("Max retries reached. Code execution failed after 5 attempts.")
            return False
        
        dataset_logger.info(f"✓ Successfully generated parsing code for {dataset_name}")
        return True
        
    except Exception as e:
        dataset_logger.error(f"Error processing dataset {dataset_name}: {e}")
        return False
    finally:
        # Close handlers
        for handler in dataset_logger.handlers:
            handler.close()
        dataset_logger.handlers = []


def main():
    parser = argparse.ArgumentParser(description="Generate parsing code for datasets")
    parser.add_argument("--index", type=str, default="output/kaggle_index.jsonl", 
                        help="Path to kaggle index JSONL file")
    parser.add_argument("--dataset", type=str, default=None, 
                        help="Specific dataset name to process (if not provided, process all)")
    parser.add_argument("--file", type=str, default=None,
                        help="Specific data file path to process")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Maximum number of datasets to process")
    args = parser.parse_args()
    
    load_env_variables()
    frozen_seed(CFG.RANDOM_SEED)
    
    gpt = setup_openai()
    chain_function = constrained_self_and_plan_chains
    
    # Setup console logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # If --dataset and --file are specified, process single dataset directly
    if args.dataset and args.file:
        logging.info(f"Processing single dataset: {args.dataset}")
        logging.info(f"Data file: {args.file}")
        success = process_dataset(args.file, args.dataset, gpt, chain_function)
        exit(0 if success else 1)
    
    # Scan kaggle_raw directory directly for all datasets
    logging.info(f"Scanning dataset directories for all datasets")
    kaggle_raw_dir = os.path.join(PROJECT_ROOT, "dataset", "kaggle_raw")
    multi_source_dir = os.path.join(PROJECT_ROOT, "dataset", "multi_source")
    
    # Support scanning multiple directories
    scan_dirs = []
    if os.path.exists(kaggle_raw_dir):
        scan_dirs.append(("kaggle_raw", kaggle_raw_dir))
    if os.path.exists(multi_source_dir):
        scan_dirs.append(("multi_source", multi_source_dir))
    
    if not scan_dirs:
        logging.error(f"No dataset directories found")
        return
    
    datasets = []
    import glob
    
    # Supported file extensions
    supported_extensions = {
        'csv': ['.csv'],
        'json': ['.json', '.jsonl'],
        'txt': ['.txt', '.log'],
        'sqlite': ['.db', '.sqlite', '.sqlite3'],
        'yaml': ['.yaml', '.yml'],
        'xml': ['.xml'],
        'parquet': ['.parquet'],
        'excel': ['.xls', '.xlsx'],
        'hdf5': ['.h5', '.hdf5'],
        'netcdf': ['.nc'],
        'avro': ['.avro'],
        'proto': ['.proto'],
        'toml': ['.toml'],
        'ini': ['.ini'],
        'conf': ['.conf', '.cfg'],
        'geo': ['.geojson', '.shp', '.kml', '.gpx'],
        'markdown': ['.md']
    }
    
    # Scan all dataset directories
    for source_name, source_dir in scan_dirs:
        logging.info(f"Scanning {source_name}: {source_dir}")
        
        # Find all directories
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            
            # Skip .temp directories and non-directories
            if item.endswith('.temp') or not os.path.isdir(item_path):
                continue
            
            # Check if this is a dataset directory (has data files directly)
            has_files = False
            for file_type, extensions in supported_extensions.items():
                for ext in extensions:
                    if glob.glob(os.path.join(item_path, f"**/*{ext}"), recursive=True):
                        has_files = True
                        break
                if has_files:
                    break
            
            if has_files:
                # Direct dataset directory
                file_types = {}
                for file_type, extensions in supported_extensions.items():
                    files = []
                    for ext in extensions:
                        pattern = os.path.join(item_path, f"**/*{ext}")
                        files.extend(glob.glob(pattern, recursive=True))
                    if files:
                        file_types[file_type] = files
                
                datasets.append({
                    'owner_slug': source_name,
                    'dataset_slug': item,
                    'dataset_ref': f"{source_name}/{item}",
                    'file_types': file_types,
                    'source_dir': source_dir
                })
            else:
                # Check if it's an owner directory with dataset subdirectories
                for dataset_dir in os.listdir(item_path):
                    dataset_path = os.path.join(item_path, dataset_dir)
                    
                    # Skip .temp directories and non-directories
                    if dataset_dir.endswith('.temp') or not os.path.isdir(dataset_path):
                        continue
                    
                    # Find all supported files in this dataset
                    file_types = {}
                    for file_type, extensions in supported_extensions.items():
                        files = []
                        for ext in extensions:
                            pattern = os.path.join(dataset_path, f"**/*{ext}")
                            files.extend(glob.glob(pattern, recursive=True))
                        if files:
                            file_types[file_type] = files
                    
                    # Add dataset if it has any supported files
                    if file_types:
                        datasets.append({
                            'owner_slug': item,
                            'dataset_slug': dataset_dir,
                            'dataset_ref': f"{item}/{dataset_dir}",
                            'file_types': file_types,
                            'source_dir': source_dir
                        })
    
    logging.info(f"Found {len(datasets)} datasets in all directories")
    
    # Apply limit if specified
    if args.limit:
        datasets = datasets[:args.limit]
        logging.info(f"Processing first {args.limit} datasets")
    
    # Process each dataset
    success_count = 0
    fail_count = 0
    
    for idx, dataset_info in enumerate(datasets, 1):
        dataset_ref = dataset_info['dataset_ref']
        file_types = dataset_info.get('file_types', {})
        source_dir = dataset_info.get('source_dir', kaggle_raw_dir)
        
        # Get all files (prioritize CSV, then JSON, then YAML/LOG, then others)
        all_files = []
        for ftype in ['csv', 'json', 'yaml', 'txt', 'xml', 'sqlite', 'parquet', 'excel', 
                      'geo', 'hdf5', 'netcdf', 'avro', 'proto', 'toml', 'ini', 'conf', 'markdown']:
            if ftype in file_types:
                all_files = file_types[ftype]
                break
        
        if not all_files:
            logging.warning(f"No supported files found for {dataset_ref}, skipping")
            continue
        
        # Use first file and fix path to current workspace
        file_path = all_files[0]
        # Fix path if it's from a different location
        if not os.path.exists(file_path):
            # Try to construct path relative to current workspace
            filename = os.path.basename(file_path)
            owner_slug = dataset_info['owner_slug']
            dataset_slug = dataset_info['dataset_slug']
            file_path = os.path.join(source_dir, owner_slug, dataset_slug, filename)
            # Also try direct path
            if not os.path.exists(file_path):
                file_path = os.path.join(source_dir, dataset_slug, filename)
        
        dataset_name = dataset_ref.replace('/', '_')
        
        # Skip if specific dataset requested and this isn't it
        if args.dataset and args.dataset != dataset_name:
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"[{idx}/{len(datasets)}] Processing: {dataset_ref}")
        logging.info(f"File: {file_path}")
        logging.info(f"{'='*60}\n")
        
        if process_dataset(file_path, dataset_name, gpt, chain_function):
            success_count += 1
        else:
            fail_count += 1
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing complete!")
    logging.info(f"✓ Success: {success_count}")
    logging.info(f"✗ Failed: {fail_count}")
    logging.info(f"{'='*60}")

if __name__ == '__main__':
    main()
