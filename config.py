import os
from dotenv import load_dotenv, find_dotenv

class CFG:
    WM = '100011100101100011100101'
    # Static initial value (will be overridden in adaptive mode)
    GAMMA = 100  # original 2
    # Target number of watermark operations to embed (approximate), used for adaptive density calculation
    TARGET_EMBED_COUNT = 20000  # Increased to 20000 to ensure low-density embedding
    # Target modification rate (prioritized if set)
    TARGET_MODIFICATION_RATE = 0.08  # 8% target modification rate (default)
    # Small datasets use higher target modification rate
    # [Strategy A] Reduced from 0.70 to 0.50, expected actual modification rate ~28% (passes 20% threshold)
    TARGET_MODIFICATION_RATE_SMALL = 0.50  # 50% target modification rate (<1000 rows, compensate for filtering loss)
    SMALL_DATASET_THRESHOLD = 1000  # Row threshold for small datasets
    # Gamma min and max value limits
    MIN_GAMMA = 2
    MAX_GAMMA = 5000  # Increased upper limit to support multi-column datasets (134 columns need ~1600 gamma)
    # Enable adaptive density (auto-calculate GAMMA based on data scale and number of index attributes)
    ADAPTIVE_GAMMA = True
    # Global tolerance adaptive scaling parameter (global_tolerance ≈ scale * expected_modification_fraction)
    TOLERANCE_SCALING = 1.8
    # Global tolerance lower and upper bounds
    GLOBAL_TOLERANCE_MIN = 0.01
    GLOBAL_TOLERANCE_MAX = 0.25
    # [Strategy B] Tiered tolerance strategy: small datasets 35%, medium 25%, large 20%
    ENABLE_TIERED_TOLERANCE = True  # Enable tiered tolerance
    TOLERANCE_SMALL = 0.35  # <1000 rows: 35%
    TOLERANCE_MEDIUM = 0.25  # 1000-10000 rows: 25%
    TOLERANCE_LARGE = 0.20  # >10000 rows: 20%
    MEDIUM_DATASET_THRESHOLD = 10000  # Medium-large dataset threshold
    RANDOM_SEED = 42
    N_SAMPLES = 5000
    CANDIDATE_P = 0.5
    DELTA = 0.01
    PERSONAL_KEY = 'example'
    MOD = 100
    KNOWLEDGE_DATABASE = {}
    # [Strategy C] Dataset quality filtering configuration
    ENABLE_DATASET_FILTERING = True  # Enable dataset filtering
    MAX_MISSING_RATE = 0.50  # Max missing value rate 50% (exclude ellimaaac 62% missing issue)
    MAX_DATASET_ROWS = 100000  # Max row limit 100k (exclude neelagiriaditya 210k extra large)
    MIN_DATASET_ROWS = 100  # Min row limit 100 (ensure statistical stability)
    MIN_NUMERIC_COLUMNS = 2  # Require at least 2 numeric columns (exclude pure text datasets like umerhaddii)
    MIN_NUMERIC_RATIO = 0.15  # Min numeric column ratio 15% (ensure sufficient numeric features)
    MAX_TEXT_RATIO = 0.80  # Max text column ratio 80% (exclude text-heavy datasets like Indonesian Hoax)
    MAX_COLUMNS = 100  # Max column limit (control processing complexity)
    MAX_FILESIZE_MB = 500  # Max file size 500MB
    # Strategy C - Exclusion list (based on data quality analysis)
    EXCLUDED_DATASETS = [
        'direct_ellimaaac',          # 62% missing values
        'ellimaaac_gpus-specs',      # 62% missing values
        'direct_neelagiriaditya',    # 210k rows extra large
        'direct_umerhaddii',         # Pure text data
        'direct_Indonesian Hoax News Dataset',  # 97.51% modification rate, text-heavy
        'direct_Coursera Courses Metadata for Analytics 2025',  # 48.14% modification rate, abnormal data features
        'direct_andrewmvd_spotify-playlists',  # 1127MB extra large file, exceeds 500MB limit
        'direct_abhi8923shriv_sentiment-analysis-dataset',  # Encoding error, cannot parse
        'direct_balaka18_email-spam-classification-dataset-csv',  # Processing time too long, may hang
    ]

def frozen_seed(seed=CFG.RANDOM_SEED):
    import random, os, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_env_variables():
    _ = load_dotenv(find_dotenv())
    # open_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["https_proxy"] = os.getenv("https_proxy")
    os.environ["http_proxy"] = os.getenv("http_proxy")

    api_key = os.getenv("API_KEY")  
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise EnvironmentError("API_KEY not found in the environment.")
    if not base_url:
        raise EnvironmentError("BASE_URL not found in the environment.")    

    return api_key, base_url
