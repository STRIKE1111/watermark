import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from pandas.api.types import CategoricalDtype

def compute_adaptive_gamma(cfg, num_rows: int, num_index_attrs: int) -> int:

    if not getattr(cfg, 'ADAPTIVE_GAMMA', False):
        return cfg.GAMMA
    
    # Select target modification rate based on data scale
    small_threshold = getattr(cfg, 'SMALL_DATASET_THRESHOLD', 1000)
    if num_rows < small_threshold:
        # Use higher target modification rate for small datasets
        target_mod_rate = getattr(cfg, 'TARGET_MODIFICATION_RATE_SMALL', 0.15)
    else:
        # Use standard target modification rate for regular datasets
        target_mod_rate = getattr(cfg, 'TARGET_MODIFICATION_RATE', 0.08)
    
    if target_mod_rate is not None and target_mod_rate > 0:
        # gamma = 1 / (1 - (1 - p)^(1/|Is|))
        num_index = max(1, num_index_attrs)
        if target_mod_rate >= 0.99:  # Avoid numerical issues
            gamma_from_rate = 1
        else:
            # (1 - p)^(1/|Is|) 
            base = (1 - target_mod_rate) ** (1.0 / num_index)
            gamma_from_rate = int(1.0 / (1.0 - base))
        
        # Set reasonable upper and lower bounds
        min_gamma = getattr(cfg, 'MIN_GAMMA', 10)
        max_gamma = getattr(cfg, 'MAX_GAMMA', 5000)  # Increase upper limit to support multi-column datasets
        gamma = max(min_gamma, min(max_gamma, gamma_from_rate))
        return gamma
    
    # Original logic (based on TARGET_EMBED_COUNT)
    target = getattr(cfg, 'TARGET_EMBED_COUNT', 1000)
    raw = int((num_rows * max(1, num_index_attrs)) / max(1, target))
    gamma = max(1, raw)
    return gamma

def custom_serializer(obj):
    if isinstance(obj, np.int64):
        return int(obj)  
    raise TypeError("Type not serializable")

def Injection(CFG, DATA, selected_attributes, domain_groups, indices, domain_gen, dataset_name="code1"):
    start_time = time.time()  

    # Use os.path to handle paths and ensure directory exists
    import os
    swap_dir = os.path.join("parser_data", dataset_name, "swap")
    os.makedirs(swap_dir, exist_ok=True)
    swap_file = os.path.join(swap_dir, "swap.json")
    swap_info = {}

    # Compute adaptive GAMMA
    adaptive_gamma = compute_adaptive_gamma(CFG, len(DATA), len(indices))
    print(f'DEBUG: Computed adaptive_gamma={adaptive_gamma} for N={len(DATA)}, |Is|={len(indices)}')
    
    modification_count = 0
    hit_count = 0
    na_count = 0
    no_group_count = 0
    small_group_count = 0

    for ex, row in tqdm(DATA.iterrows(), total=len(DATA)):
        for idx_pos, index in enumerate(indices):
            # 使用索引位置作为extra_param确保每个索引列独立生成随机数
            random_values = domain_gen.generate_seed(CFG.PERSONAL_KEY, row.iloc[index], extra_param=idx_pos)
            if random_values[0] % adaptive_gamma == 0:
                hit_count += 1
                i = random_values[1] % len(selected_attributes)
                j = random_values[2] % len(CFG.WM)
                v = row[selected_attributes[i]]
                if pd.isna(v):
                    na_count += 1
                    continue
                group = domain_gen.find_closest_value(v, domain_groups[selected_attributes[i]])
                if group is None:
                    no_group_count += 1
                    continue
                if len(group) <= 1:
                    small_group_count += 1
                    continue
                k = domain_gen.Hash(CFG.PERSONAL_KEY, row.iloc[index], CFG.WM[j]) % len(group)
                # print(f'Modifying row {ex}, column {selected_attributes[i]}: {v} -> {list(group)[k]}')

                if isinstance(DATA[selected_attributes[i]].dtype, CategoricalDtype) and list(group)[k] not in DATA[selected_attributes[i]].cat.categories:
                    DATA[selected_attributes[i]] = DATA[selected_attributes[i]].cat.add_categories(list(group)[k])
                DATA.at[ex, selected_attributes[i]] = list(group)[k]

                swap_info[ex] = [v, list(group)[k]]
                modification_count += 1

    print(f'DEBUG: Modified {modification_count} cells across {len(swap_info)} rows')
    print(f'DEBUG: Stats - Hits:{hit_count}, NA:{na_count}, NoGroup:{no_group_count}, SmallGroup:{small_group_count}')
    print('Injection completed.\n')
    end_time = time.time()  

    with open(swap_file, 'w') as f:
        json.dump(swap_info, f, default=custom_serializer)

    total_time = end_time - start_time  
    return DATA, total_time


def modify_dataframe(df, modify_percent):
    df_modified = df.copy()
    num_rows = len(df)
    num_modify = int(num_rows * modify_percent / 100)
    rows_to_modify = np.random.choice(df.index, size=num_modify, replace=False)
    for row in rows_to_modify:
        # Select non-boolean columns for modification
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        category_cols = df.select_dtypes(include=['object', 'category']).columns
        valid_cols = list(numeric_cols) + list(category_cols)
        
        if not valid_cols:
            print("Warning: No suitable columns found for modification")
            continue
            
        col_to_modify = np.random.choice(valid_cols)
        col_type = df[col_to_modify].dtype
        
        if col_type in ['int64', 'float64']:
            random_value = np.random.choice(np.arange(1, 10000))
        else:
            # For category or string types, select from existing unique values
            unique_values = df[col_to_modify].unique()
            if len(unique_values) > 0:
                random_value = np.random.choice(unique_values)
            else:
                continue
                
        df_modified.at[row, col_to_modify] = random_value
    
    return df_modified


def Detection(CFG, watermarked_data, selected_attributes, domain_groups, indices, domain_gen):
    DATA = pd.read_csv(watermarked_data)
    # Reduce modify_percent to speed up detection
    DATA = modify_dataframe(DATA, modify_percent=10)
    
    num_bits = len(CFG.WM)
    Count = np.zeros((num_bits, 2))
    WM_x = np.zeros(num_bits)
    start_time = time.time()
    
    # Use the same adaptive GAMMA calculation logic as injection phase
    adaptive_gamma = compute_adaptive_gamma(CFG, len(DATA), len(indices))

    for i, row in tqdm(DATA.iterrows(), total=len(DATA)):
        for idx_pos, index in enumerate(indices):
            # print(f'index: {index}, row.iloc[index]: {row.iloc[index]}')
            # 使用索引位置作为extra_param确保每个索引列独立生成随机数
            random_values = domain_gen.generate_seed(CFG.PERSONAL_KEY, row.iloc[index], extra_param=idx_pos)
            if random_values[0] % adaptive_gamma == 0:
                i = random_values[1] % len(selected_attributes)
                j = random_values[2] % len(CFG.WM)
                v = row[selected_attributes[i]]
                if not pd.isna(v):
                    group = domain_gen.find_closest_value(v, domain_groups[selected_attributes[i]])
                    # print(seed_values, i, j, v, group)
                    if group is not None and len(group) > 1:
                        k = domain_gen.Hash(CFG.PERSONAL_KEY, row.iloc[index], CFG.WM[j]) % len(group)
                        if list(group)[k] == v:
                            Count[j][eval(CFG.WM[j])] += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    # print(f"Total detection time: {total_time:.2f} seconds")
    for b in range(num_bits):
            if Count[b][1] > Count[b][0]:
                WM_x[b] = 1
            elif Count[b][1] < Count[b][0]:
                WM_x[b] = 0
            else:
                print(f'watermark_x[{b}] not detected.')
    
    WM = np.array([eval(bit) for bit in CFG.WM])  
    correct_bits = np.sum(WM == WM_x)
    print(Count)  
    accuracy = (correct_bits / num_bits) * 100

    print(f"Watermark detection accuracy: {accuracy:.2f}%")
    
    return WM_x, accuracy, total_time