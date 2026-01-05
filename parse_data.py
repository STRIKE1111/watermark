import logging
import random
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse

# ensure project root is on sys.path so local modules import reliably
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import CFG, frozen_seed, load_env_variables
from data_processing import (
    DataProcessor,
    FeatureSelector,
    DomainProcessor,
)
from watermark import (
    Injection,
    Detection
)
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)     
pd.set_option('display.width', 1000)

def process_single_dataset(input_filepath, dataset_name, api_key, base_url):
    """Process a single dataset: parse, add watermark, and verify."""
    
    dataset_dir = os.path.join("parser_data", dataset_name, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Copy original dataset to processing directory (if not in same location)
    import shutil
    output_filepath = os.path.join(dataset_dir, os.path.basename(input_filepath))
    
    # Only copy if source and destination are different
    input_abs = os.path.abspath(input_filepath)
    output_abs = os.path.abspath(output_filepath)
    if input_abs != output_abs:
        shutil.copy2(input_filepath, output_filepath)
    
    # Use processed file path
    filepath = output_filepath

    base_filename = os.path.basename(filepath)
    log_filename = f"{base_filename.split('.')[0]}.log"
    log_folder = os.path.join('parser_data', dataset_name, 'en_de_time_Log')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Create dataset-specific logger
    dataset_logger = logging.getLogger(dataset_name)
    dataset_logger.setLevel(logging.INFO)
    dataset_logger.handlers = []
    
    fh = logging.FileHandler(os.path.join(log_folder, log_filename), mode='a+', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    dataset_logger.addHandler(fh)

    try:
        dataset_logger.info(f'Dataset name: {base_filename}')
        
        # Check file size and estimated row count early
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        if file_size_mb > CFG.MAX_FILESIZE_MB:
            dataset_logger.warning(f'File too large: {file_size_mb:.2f}MB > {CFG.MAX_FILESIZE_MB}MB, skipping')
            print(f'[{dataset_name}] Skipped: File too large ({file_size_mb:.2f}MB)')
            return False
        
        processor = DataProcessor(filepath)
        DATA, parse_time = processor.get_structured_data()
        
        # Check row limit
        if len(DATA) > CFG.MAX_DATASET_ROWS:
            dataset_logger.warning(f'Dataset too large: {len(DATA)} rows > {CFG.MAX_DATASET_ROWS} rows, skipping')
            print(f'[{dataset_name}] Skipped: Too many rows ({len(DATA)} > {CFG.MAX_DATASET_ROWS})')
            return False
        
        # Handle columns with list/dict types (convert to string or skip)
        cols_to_drop = []
        for col in DATA.columns:
            # Check if column contains unhashable types (list, dict, etc.)
            sample_values = DATA[col].dropna().head(10)
            has_unhashable = False
            for val in sample_values:
                if isinstance(val, (list, dict, set)):
                    has_unhashable = True
                    break
            if has_unhashable:
                # Try converting to string
                try:
                    DATA[col] = DATA[col].apply(lambda x: str(x) if isinstance(x, (list, dict, set)) else x)
                    dataset_logger.info(f'Converted unhashable column to string: {col}')
                except Exception:
                    cols_to_drop.append(col)
                    dataset_logger.warning(f'Dropping column with unhashable types: {col}')
        
        if cols_to_drop:
            DATA = DATA.drop(columns=cols_to_drop)
            print(f'[{dataset_name}] Dropped {len(cols_to_drop)} columns with complex types')
        
        # Remove emoji and zero-width characters from column names
        import re
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002300-\U000023FF"  # Miscellaneous Technical
            u"\u200B-\u200D"  # Zero Width Space/Joiner
            u"\uFEFF"  # Zero Width No-Break Space
            "]+", flags=re.UNICODE)
        DATA.columns = [emoji_pattern.sub('', col).strip() for col in DATA.columns]
        
        # Fix empty column name issue: rename empty string column names to unnamed_<index>
        new_columns = []
        for i, col in enumerate(DATA.columns):
            if col.strip() == '':
                new_col = f'unnamed_{i}'
                dataset_logger.warning(f'Empty column name at index {i}, renamed to: {new_col}')
                new_columns.append(new_col)
            else:
                new_columns.append(col)
        DATA.columns = new_columns
        
        # Remove all-null columns and high-null columns (>80% null)
        null_ratios = DATA.isnull().sum() / len(DATA)
        high_null_cols = null_ratios[null_ratios > 0.8].index.tolist()
        if high_null_cols:
            DATA = DATA.drop(columns=high_null_cols)
            dataset_logger.info(f'Dropped {len(high_null_cols)} columns with >80% null values: {high_null_cols[:5]}...')
            print(f'[{dataset_name}] Dropped {len(high_null_cols)} high-null columns')
        
        # If no columns left after removal, skip
        if len(DATA.columns) == 0:
            dataset_logger.warning('No columns left after removing null columns')
            print(f'[{dataset_name}] SKIPPED: No valid columns after removing null columns')
            return False
        
        # Check data size: filter datasets that are too small
        if len(DATA) < 50:
            dataset_logger.warning(f'⚠️ Dataset too small: {len(DATA)} rows (minimum: 50)')
            dataset_logger.warning('Skipping watermark injection for small dataset')
            print(f'[{dataset_name}] Skipped: Data too small ({len(DATA)} rows)')
            return False
        elif len(DATA) < 100:
            dataset_logger.warning(f'⚠️ Small dataset: {len(DATA)} rows, proceeding with adjusted parameters')
            print(f'[{dataset_name}] Warning: Small dataset ({len(DATA)} rows), adjusting parameters')
        
        # Recalculate overall null ratio (after removing high-null columns)
        overall_null_ratio = DATA.isnull().sum().sum() / (len(DATA) * len(DATA.columns)) if len(DATA.columns) > 0 else 1.0
        if overall_null_ratio > 0.7:  # Increased threshold to 70%
            dataset_logger.warning(f'⚠️ Dataset has too many null values: {overall_null_ratio:.1%}')
            dataset_logger.warning('Skipping watermark injection due to high null ratio')
            print(f'[{dataset_name}] Skipped: Too many null values ({overall_null_ratio:.1%})')
            return False
        elif overall_null_ratio > 0.3:
            dataset_logger.warning(f'⚠️ Dataset has significant null values: {overall_null_ratio:.1%}, proceeding cautiously')
            print(f'[{dataset_name}] Warning: Significant null values ({overall_null_ratio:.1%})')
        
        # Safe print to avoid Unicode encoding errors
        try:
            print(f'[{dataset_name}] Data head 2: \n{DATA.head(2)}')
        except UnicodeEncodeError:
            print(f'[{dataset_name}] Data head 2: [Unicode display error - skipped]')
        print(f"[{dataset_name}] Data info:\n{DATA.info()}")
        
        csv_dir = os.path.join('parser_data', dataset_name, 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        # Increase float precision when saving to avoid watermark information loss
        DATA.to_csv(os.path.join(csv_dir, f'original_{base_filename.split(".")[0]}.csv'), 
                    index=False, encoding='utf-8-sig', float_format='%.10f')
        
        dataset_logger.info(f'DATA length: {len(DATA)}')
        dataset_logger.info(f'Parse time: {parse_time}')
        DATA = processor.str2num(DATA)    
        DATA = processor.str2category(DATA)

        # Smaller sampling to speed up feature selection (0.1% or max 1000 rows)
        sample_size = min(1000, max(10, int(len(DATA) * 0.001)))
        data_sample = DATA.sample(n=sample_size, random_state=CFG.RANDOM_SEED)
        
        # Calculate missing ratio per column, skip columns with >20% missing values
        missing_threshold = 0.20
        non_null_cols = [col for col in data_sample.columns 
                        if data_sample[col].notnull().sum() >= len(data_sample) * (1 - missing_threshold)]
        
        missing_info = {col: (1 - data_sample[col].notnull().sum() / len(data_sample)) 
                       for col in data_sample.columns if col not in non_null_cols}
        if missing_info:
            dataset_logger.info(f'Skipped columns with >20% missing values: {missing_info}')
            print(f'[{dataset_name}] Skipped {len(missing_info)} columns with high missing rate')
        
        selector = FeatureSelector()
        valid_fields = selector.get_numeric_category_cols(data_sample)
        print(f'[{dataset_name}] Initial valid_fields: {valid_fields}')
        
        valid_fields1 = selector.get_candidate_indices1(data_sample, non_null_cols, P=4)
        print(f'[{dataset_name}] Valid fields1: {valid_fields1}')
        dataset_logger.info(f'valie_fields1: {valid_fields1}')

        valid_fields2 = selector.get_candidate_indices2(data_sample, valid_fields, CFG.CANDIDATE_P)
        print(f'[{dataset_name}] Initial valid_fields2 (before sampling): {valid_fields2}')
        
        if len(valid_fields2) >= 2:
            valid_fields2 = set(random.sample(list(valid_fields2), 2))
        else:
            valid_fields2 = set()
            print(f'[{dataset_name}] Warning: valid_fields2 is empty or has less than 2 elements')
        
        print(f'[{dataset_name}] Final valid_fields2 (after sampling): {valid_fields2}')
        dataset_logger.info(f'valie_fields2: {valid_fields2}')

        location_fields = list(valid_fields1 | valid_fields2)
        print(f'[{dataset_name}] Combined location_fields: {location_fields}')
        
        if not location_fields:
            print(f'[{dataset_name}] SKIPPED: No location fields selected - dataset may have too many null values or unsuitable columns')
            dataset_logger.warning('Skipping injection: No location fields selected (null values or unsuitable columns)')
            return False
            
        dataset_logger.info(f'location_fields:{location_fields}')

        # 确保Indices是Python整数类型，而不是numpy整数
        Indices = [int(idx) for idx in random.sample([data_sample.columns.to_list().index(col) for col in location_fields], 
                                min(4, max(len(valid_fields1), len(valid_fields2))))]
        dataset_logger.info(f'Indices:{Indices}')
        selected_cols = [data_sample.columns.to_list()[int(col)] for col in Indices]
        nunique_values = DATA[selected_cols].nunique()

        dataset_logger.info(f"Selected columns: {selected_cols}")
        dataset_logger.info(f"Unique value counts:\n{nunique_values}")

        Attributes = selector.get_candidate_attributes(data_sample, valid_fields)  
        print(f'[{dataset_name}] Selected Attributes: {Attributes}')
        
        if not Attributes:
            print(f'[{dataset_name}] SKIPPED: No suitable attributes for watermarking')
            dataset_logger.warning('Skipping injection: No suitable attributes selected')
            return False
        
        # 只保留数值类型的属性，过滤掉字符串列
        numeric_cols = DATA.select_dtypes(include=[np.number]).columns.tolist()
        Attributes_filtered = [attr for attr in Attributes if attr in numeric_cols]
        
        if len(Attributes_filtered) < len(Attributes):
            removed = set(Attributes) - set(Attributes_filtered)
            dataset_logger.warning(f'Removed non-numeric attributes: {removed}')
            print(f'[{dataset_name}] Filtered out non-numeric attributes: {removed}')
        
        # 如果没有数值属性，尝试从字符串列创建长度特征
        if len(Attributes_filtered) == 0:
            string_cols = DATA.select_dtypes(include=['object']).columns.tolist()
            if string_cols:
                # 创建字符串长度列作为数值特征
                for col in string_cols[:3]:  # 最多取3个字符串列
                    len_col = f'{col}_len'
                    DATA[len_col] = DATA[col].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
                    Attributes_filtered.append(len_col)
                    dataset_logger.info(f'Created string length feature: {len_col}')
                print(f'[{dataset_name}] Created {len(Attributes_filtered)} string length features')
                numeric_cols = DATA.select_dtypes(include=[np.number]).columns.tolist()
        
        # 进一步过滤掉缺失值过多的属性（全量数据检查）
        missing_threshold = 0.30  # 提高到30%
        Attributes_no_missing = [attr for attr in Attributes_filtered 
                                if DATA[attr].notnull().sum() >= len(DATA) * (1 - missing_threshold)]
        
        if len(Attributes_no_missing) < len(Attributes_filtered):
            removed_missing = set(Attributes_filtered) - set(Attributes_no_missing)
            missing_rates = {attr: f"{(1 - DATA[attr].notnull().sum() / len(DATA)):.1%}" 
                           for attr in removed_missing}
            dataset_logger.warning(f'Removed attributes with >20% missing: {missing_rates}')
            print(f'[{dataset_name}] Filtered out {len(removed_missing)} attributes with high missing rate: {missing_rates}')
        
        if len(Attributes_no_missing) == 0:
            dataset_logger.warning('⚠️ Skipping: No numeric attributes available for watermarking (after missing value filter)')
            print(f'[{dataset_name}] SKIPPED: No valid numeric attributes without excessive missing values')
            return False
        
        Attributes = Attributes_no_missing
        dataset_logger.info(f'Attributes (numeric only): {Attributes}')
        
        domain_gen = DomainProcessor(DATA, Attributes, CFG.DELTA)
        print(f'[{dataset_name}] Domain generating...')
        domain_groups = domain_gen.generate_domain(api_key, base_url)
        
        output_dir = os.path.join('parser_data', dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'output.txt'), 'w', encoding='utf-8') as txt_file:
            txt_file.write(str(domain_groups))

        # 只添加一个Index列（不使用reset_index避免重复）
        DATA['Index'] = range(len(DATA))
        
        # 确保Indices中的每个元素都是Python整数
        Indices = [int(index) + 1 for index in Indices]
        Indices.append(0)
        print(f'[{dataset_name}] Indices:{Indices}')
        cols = [DATA.columns.to_list()[int(index)] for index in Indices]
        print(f'[{dataset_name}] Cols:{cols}')

        print(f'[{dataset_name}] Watermark injecting...')
        watermarked_data, injection_time = Injection(CFG, DATA, Attributes, domain_groups, Indices, domain_gen, dataset_name)
        
        statis_dir = os.path.join("parser_data", dataset_name, "statis")
        os.makedirs(statis_dir, exist_ok=True)
        
        watermarked_data_path = os.path.join(statis_dir, f'watermark_{base_filename.split(".")[0]}.csv')
        watermarked_data.to_csv(watermarked_data_path, index=False, encoding='utf-8-sig')

        dataset_logger.info(f'Injection time: {injection_time}')
        
        print(f'[{dataset_name}] Watermark detecting...')
        
        with open(os.path.join(output_dir, 'output.txt'), 'r', encoding='utf-8') as txt_file:
            content = txt_file.read()
            # 添加inf支持，避免NameError
            domain_groups = eval(content, {
                'np': np, 
                'nan': float('nan'),
                'inf': float('inf'),
                'Infinity': float('inf'),
                '-inf': float('-inf'),
                '-Infinity': float('-inf')
            }, {})
        
        # 采样验证数据以加速Detection（最多5000行）
        detection_sample_path = watermarked_data_path
        if len(watermarked_data) > 5000:
            detection_data = watermarked_data.sample(n=5000, random_state=CFG.RANDOM_SEED)
            detection_sample_path = os.path.join(statis_dir, f'watermark_{base_filename.split(".")[0]}_sample.csv')
            detection_data.to_csv(detection_sample_path, index=False, encoding='utf-8-sig')
            print(f'[{dataset_name}] Using sampled data for detection: 5000/{len(watermarked_data)} rows')
            
        WM_x, accuracy, detection_time = Detection(CFG, detection_sample_path, Attributes, domain_groups, Indices, domain_gen)
        
        dataset_logger.info(f'WM_x: {WM_x}')
        dataset_logger.info(f'Accuracy: {accuracy}')
        dataset_logger.info(f'Detection time: {detection_time}')
        
        print(f'[{dataset_name}] Successfully processed')
        return True
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        dataset_logger.error(f'Error processing dataset: {e}')
        dataset_logger.error(f'Traceback:\n{error_trace}')
        # 安全打印错误信息
        try:
            print(f'[{dataset_name}] Failed: {str(e)}')
        except UnicodeEncodeError:
            print(f'[{dataset_name}] Failed: [Unicode error]')
        try:
            print(f'Traceback: {error_trace}')
        except UnicodeEncodeError:
            print(f'[{dataset_name}] Traceback contains unicode characters')
        return False
    finally:
        for handler in dataset_logger.handlers:
            handler.close()
        dataset_logger.handlers = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add watermark to datasets")
    parser.add_argument("--index", type=str, default="output/kaggle_index.jsonl", 
                        help="Path to kaggle index JSONL file")
    parser.add_argument("--dataset", type=str, default=None, 
                        help="Specific dataset name to process")
    parser.add_argument("--file", type=str, default=None,
                        help="Specific data file path to process")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Maximum number of datasets to process")
    args = parser.parse_args()
    
    api_key, base_url = load_env_variables()
    frozen_seed(CFG.RANDOM_SEED)
    
    # Setup console logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # If --dataset and --file are specified, process single dataset directly
    if args.dataset and args.file:
        logging.info(f"Processing single dataset: {args.dataset}")
        logging.info(f"Data file: {args.file}")
        success = process_single_dataset(args.file, args.dataset, api_key, base_url)
        exit(0 if success else 1)
    
    # Scan generated_code directory for all convert2json scripts
    logging.info(f"Scanning generated_code directory for datasets")
    generated_code_dir = os.path.join(PROJECT_ROOT, "generated_code")
    
    if not os.path.exists(generated_code_dir):
        logging.error(f"generated_code directory not found: {generated_code_dir}")
        exit(1)
    
    import glob
    
    # Find all convert2json_*.py files
    convert_scripts = glob.glob(os.path.join(generated_code_dir, "convert2json_*.py"))
    
    # Define data source directories
    kaggle_raw_dir = os.path.join(PROJECT_ROOT, "dataset", "kaggle_raw")
    multi_source_dir = os.path.join(PROJECT_ROOT, "dataset", "multi_source")
    
    # Supported file extensions
    supported_extensions = ['*.csv', '*.json', '*.xml', '*.yaml', '*.yml', '*.db', '*.sqlite', '*.sqlite3', '*.parquet', '*.txt', '*.log']
    
    # Extract dataset names from script filenames
    datasets = []
    for script in convert_scripts:
        script_name = os.path.basename(script)
        # Extract dataset name: convert2json_{dataset_name}.py
        dataset_name = script_name.replace('convert2json_', '').replace('.py', '')
        
        # Try to find the data file
        data_file = None
        dataset_path = None
        
        # 首先检查multi_source目录（处理github_xxx等数据集）
        if dataset_name.startswith('multi_source_'):
            actual_name = dataset_name.replace('multi_source_', '')
            dataset_path = os.path.join(multi_source_dir, actual_name)
        elif os.path.exists(os.path.join(multi_source_dir, dataset_name)):
            dataset_path = os.path.join(multi_source_dir, dataset_name)
        elif dataset_name.startswith('direct_'):
            # Direct dataset in kaggle_raw
            actual_name = dataset_name.replace('direct_', '')
            dataset_path = os.path.join(kaggle_raw_dir, actual_name)
        elif dataset_name.startswith('kaggle_raw_'):
            actual_name = dataset_name.replace('kaggle_raw_', '')
            dataset_path = os.path.join(kaggle_raw_dir, actual_name)
        else:
            # owner/dataset structure in kaggle_raw
            parts = dataset_name.split('_', 1)
            if len(parts) == 2:
                owner, ds_name = parts
                dataset_path = os.path.join(kaggle_raw_dir, owner, ds_name)
                # 也尝试直接在multi_source中查找
                if not os.path.exists(dataset_path):
                    dataset_path = os.path.join(multi_source_dir, dataset_name)
        
        # 查找数据文件
        if dataset_path and os.path.exists(dataset_path):
            for ext in supported_extensions:
                files = glob.glob(os.path.join(dataset_path, "**", ext), recursive=True)
                if files:
                    data_file = files[0]
                    break
        
        if data_file and os.path.exists(data_file):
            datasets.append({
                'dataset_name': dataset_name,
                'data_file': data_file
            })
    
    logging.info(f"Found {len(datasets)} datasets with generated parsers")
    
    # Apply limit if specified
    if args.limit:
        datasets = datasets[:args.limit]
        logging.info(f"Processing first {args.limit} datasets")
    
    # Process each dataset
    success_count = 0
    fail_count = 0
    
    for idx, dataset_info in enumerate(datasets, 1):
        dataset_name = dataset_info['dataset_name']
        data_file = dataset_info['data_file']
        
        # Skip if specific dataset requested and this isn't it
        if args.dataset and args.dataset != dataset_name:
            continue
        
        # Skip if in excluded list
        if dataset_name in CFG.EXCLUDED_DATASETS:
            logging.info(f"[{idx}/{len(datasets)}] Skipping {dataset_name} (in excluded list)")
            continue
        
        # Skip if already completed (has swap.json)
        swap_file = os.path.join("parser_data", dataset_name, "swap", "swap.json")
        if os.path.exists(swap_file):
            logging.info(f"[{idx}/{len(datasets)}] Skipping {dataset_name} (already completed)")
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"[{idx}/{len(datasets)}] Processing: {dataset_name}")
        logging.info(f"Data file: {data_file}")
        logging.info(f"{'='*60}\n")
        
        if process_single_dataset(data_file, dataset_name, api_key, base_url):
            success_count += 1
        else:
            fail_count += 1
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing complete!")
    logging.info(f"✓ Success: {success_count}")
    logging.info(f"✗ Failed: {fail_count}")
    logging.info(f"{'='*60}")