# data_processing.py
import pandas as pd
import numpy as np
from generated_code.convert2json import parseCode
from file_parsers import read_file
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import hashlib
import random
import ast
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def get_structured_data(self):
        start_time = time.time()
        
        # Check file type
        file_extension = self.file_path[self.file_path.rfind('.'):].lower()
        
        if file_extension == '.csv':
            # For CSV files, read directly as DataFrame
            objects = read_file(self.file_path)
            if objects and isinstance(objects, list) and isinstance(objects[0], dict):
                # CSV is already a list of dicts, convert directly to DataFrame
                structured_data = pd.DataFrame(objects)
            else:
                # If not expected format, use pandas to read directly
                structured_data = pd.read_csv(self.file_path)
        else:
            # For other file types, use original parsing logic
            objects = read_file(self.file_path)
            df = pd.DataFrame(objects, columns=['objects'])
            df = df.reset_index().rename(columns={'index': 'Index'})
            df['formatted_objects'] = df['objects'].apply(lambda x: {"data": x})
            df['json_objects'] = df['formatted_objects'].apply(parseCode)
            structured_data = pd.json_normalize(df['json_objects'])
        
        end_time = time.time()
        total_time = end_time - start_time
        return structured_data, total_time

    def str2num(self, df):
        for column in df.columns:
            if df[column].dtype == 'object':
                try:
                    numeric_series = pd.to_numeric(df[column], errors='coerce')
                    if not numeric_series.isna().all():  # If not all conversion failed
                        df[column] = numeric_series
                except Exception as e:
                    print(f"Warning: Column '{column}' could not be converted to numeric. Error: {e}")
        return df

    def str2category(self, df):
        for column in df.select_dtypes(include=['object']).columns:
            try:
                if df[column].apply(lambda x: isinstance(x, (list, tuple))).any():
                    continue
                unique_count = df[column].nunique()
                if 3 <= unique_count < 50:
                    df[column] = df[column].astype('category')
            except Exception as e:
                print(f"Column '{column}' skipped due to error: {e}")
        return df

class FeatureSelector:
    @staticmethod
    def get_numeric_category_cols(df):
        categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        result = categorical_cols + numerical_cols
        if not result:
            print('Warning: No numeric or categorical columns found in the data')
            # If no numeric or categorical columns, try converting potential numeric columns
            for col in df.columns:
                try:
                    if pd.to_numeric(df[col], errors='coerce').notna().any():
                        numerical_cols.append(col)
                except:
                    continue
            result = categorical_cols + list(set(numerical_cols))
            
        return result

    @staticmethod
    def get_candidate_indices1(df, valid_fields, P):
        # Ensure valid_fields is not empty
        if not valid_fields:
            print('Warning: No valid fields provided for candidate selection')
            return set()
            
        # Remove columns with too many null values
        valid_fields = [field for field in valid_fields if df[field].notna().sum() / len(df) > 0.5]
        
        if not valid_fields:
            print('Warning: All fields have too many null values')
            return set()
            
        sorted_fields = df[valid_fields].nunique().sort_values(ascending=False).index.to_list()
        
        # Keep at least one field
        if isinstance(P, int):
            P = max(1, P)
            valid_fields1 = set(sorted_fields[:P])
        elif isinstance(P, float) and 0 < P <= 1:
            count = max(1, int(len(sorted_fields) * P))
            valid_fields1 = set(sorted_fields[:count])
        else:
            raise ValueError("P must be a float (0 < P <= 1) or an integer.")

        return valid_fields1

    @staticmethod
    def get_candidate_indices2(df, valid_cols, P):
        if not valid_cols:
            print('Warning: No valid columns provided for mutual information calculation')
            return set()
            
        # Remove columns with too many null values
        valid_cols = [col for col in valid_cols if df[col].notna().sum() / len(df) > 0.5]
        
        if len(valid_cols) < 2:
            print('Warning: Not enough valid columns for mutual information calculation')
            return set()
            
        mi_matrix = pd.DataFrame(index=valid_cols, columns=valid_cols)
        categorical_cols = df.select_dtypes(include=['category']).columns.tolist()

        for col_x in valid_cols:
            for col_y in valid_cols:
                if col_x == col_y:
                    mi_matrix.loc[col_x, col_y] = 0
                else:
                    try:
                        x_data = df[col_x].fillna(df[col_x].mean() if df[col_x].dtype.kind in 'if' else df[col_x].mode()[0])
                        y_data = df[col_y].fillna(df[col_y].mean() if df[col_y].dtype.kind in 'if' else df[col_y].mode()[0])
                        
                        if col_x in categorical_cols or col_y in categorical_cols:
                            if col_y in categorical_cols:
                                mi = mutual_info_classif(x_data.values.reshape(-1, 1), y_data.astype('category'))
                            else:
                                mi = mutual_info_classif(y_data.values.reshape(-1, 1), x_data.astype('category'))
                        else:
                            mi = mutual_info_regression(x_data.values.reshape(-1, 1), y_data)

                        mi_value = mi[0] if isinstance(mi, (list, np.ndarray)) else mi
                        mi_matrix.loc[col_x, col_y] = mi_value
                        mi_matrix.loc[col_y, col_x] = mi_value  
                    except Exception as e:
                        print(f"Warning: Error calculating MI for '{col_x}' and '{col_y}': {e}")
                        mi_matrix.loc[col_x, col_y] = 0
                        mi_matrix.loc[col_y, col_x] = 0

        mi_matrix = mi_matrix.fillna(0).copy()
        positive_counts = mi_matrix.sum()
        sorted_positive_counts = positive_counts.sort_values(ascending=False)
        # valid_fileds2 = set(sorted_positive_counts[:int(len(sorted_positive_counts) * P)].index.to_list())
        # valid_fileds2 = set(sorted_positive_counts[:4].index.to_list())

        if isinstance(P, int):
            valid_fields2 = set(sorted_positive_counts[:P].index.to_list())
        elif isinstance(P, float) and 0 < P <= 1:
            valid_fields2 = set(sorted_positive_counts[:int(len(sorted_positive_counts) * P)].index.to_list())
        else:
            raise ValueError("P must be a float (0 < P <= 1) or an integer.")


        return valid_fields2

    

    @staticmethod
    def get_candidate_attributes(sampled_data, valid_fields, max_fields=10):
        if not valid_fields:
            print('Warning: No valid fields provided for attribute selection')
            # Try to get available numeric and categorical columns from data
            numeric_columns = sampled_data.select_dtypes(include=[int, float]).columns
            category_columns = sampled_data.select_dtypes(include=['category']).columns
            
            if len(numeric_columns) == 0 and len(category_columns) == 0:
                # Try converting potential numeric columns to numeric type
                potential_numeric = []
                for col in sampled_data.columns:
                    try:
                        if pd.to_numeric(sampled_data[col], errors='coerce').notna().any():
                            potential_numeric.append(col)
                    except:
                        continue
                if potential_numeric:
                    print(f'Found {len(potential_numeric)} potential numeric columns')
                    return potential_numeric[:max_fields]
                return []
        
        # Remove columns with too many null values
        valid_fields = [field for field in valid_fields 
                       if isinstance(field, str) and  # Ensure field is string
                       field in sampled_data.columns and  # Ensure field exists in data
                       sampled_data[field].notna().sum() / len(sampled_data) > 0.5]  # Ensure enough non-null values
                       
        if not valid_fields:
            print('Warning: All fields have too many null values')
            return []
            
        if len(valid_fields) >= max_fields:
            numeric_columns = sampled_data.select_dtypes(include=[int, float]).columns
            if len(numeric_columns) >= max_fields:
                unique_counts = sampled_data[numeric_columns].nunique().sort_values(ascending=False)
                Attributes = unique_counts.index[:max_fields].tolist()
            else:
                category_columns = sampled_data.select_dtypes(include=['category']).columns
                # Fix index issue: use .tolist() to convert to list before indexing
                sorted_indices = sampled_data[category_columns].notnull().sum().sort_values().index.tolist()
                sorted_category_columns = [col for col in category_columns if col in sorted_indices]
                Attributes = list(numeric_columns) + sorted_category_columns[:(max_fields - len(numeric_columns))]
        else:
            Attributes = valid_fields
        
        # Ensure all elements in Attributes are Python string type
        Attributes = [str(attr) for attr in Attributes if attr is not None]
            
        # Ensure at least one attribute is returned
        if not Attributes and valid_fields:
            Attributes = [str(valid_fields[0])]
            
        return Attributes


class DictOutputParser(BaseOutputParser):
    def parse(self, text: str) -> set:
        # print(text)
        result_dict = ast.literal_eval(text.strip())
        # result_set = set(result_dict.keys())
        return result_dict

class DomainProcessor:
    def __init__(self, df, candidate_keys, delta):
        self.df = df
        self.candidate_keys = candidate_keys
        self.delta = delta

    def synonyms_set(self, data, api_key, base_url):
        synonyms_gen_template = """
        Now you are an expert in generating synonyms. I will provide you with data in the form of a Python list. 
        For each element in the list, you need to generate a set of exactly 3-5 synonyms, including the original element in the set. 
        If the original word is in all uppercase, ensure the generated synonyms are also in all uppercase. 
        Maintain the same case sensitivity as the original words (e.g., if the original word is capitalized, the synonyms should also be capitalized if applicable). 
        Additionally, ensure that the synonyms are contextually appropriate and maintain the same meaning as the original words. 
        The final return format should be a dictionary with the original element as the key, and a set of synonyms (including the original element) as the value. 
        **Do not include any code block markers such as triple backticks (` ``` `) or language identifiers like `python` in the response.** 
        Return only the result in this format: {{element: {{synonym1, synonym2, synonym3, synonym4, synonym5}}, ...}}.
        Ensure that all strings are properly escaped and enclosed in double quotes to avoid parsing errors.
        data: {data}
        """
        
        model_4_mini = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8, top_p=0.95, api_key=api_key, base_url=base_url)

        synonyms_gen_prompt = PromptTemplate.from_template(synonyms_gen_template)
        synonyms_gen_chain = synonyms_gen_prompt | model_4_mini | DictOutputParser()
        final_result = synonyms_gen_chain.invoke({"data": data})
        return final_result


    def dp_grouping(self, partition):
        """
        对一个分区使用动态规划分组，最小化组内差异总和。
        """
        n = len(partition)
        if n < 2:
            return [partition]  
        
        
        dp = [float('inf')] * n 
        split_point = [-1] * n  
        dp[0] = float('inf')  
        dp[1] = partition[1] - partition[0]  
        
        
        for i in range(2, n):
            for j in range(i - 1):  
                cost = dp[j] + (partition[i] - partition[j + 1]) 
                if cost < dp[i]:
                    dp[i] = cost
                    split_point[i] = j
        
        
        groups = []
        i = n - 1
        while i > 0:
            start = split_point[i] + 1
            groups.append(partition[start:i + 1])
            i = split_point[i]
        return groups[::-1]  


    def generate_domain(self, api_key, base_url):
        partitions = {}
        for key in self.candidate_keys:
            if not isinstance(key, str):
                print(f"Warning: Invalid key type: {type(key)}")
                continue
                
            if key not in self.df.columns:
                print(f"Warning: Key {key} not found in dataframe")
                continue
                
            partitions[key] = []
            
            # Handle boolean type
            if pd.api.types.is_bool_dtype(self.df[key]):
                partitions[key] = {'True': {True}, 'False': {False}}
                continue
                
            # Handle categorical type
            if isinstance(self.df[key].dtype, pd.CategoricalDtype):
                try:
                    unique_values = self.df[key].dropna().unique().tolist()
                    if unique_values:
                        partitions[key] = self.synonyms_set(unique_values, api_key=api_key, base_url=base_url)
                except Exception as e:
                    print(f"Error generating synonyms for {key}: {e}")
                    partitions[key] = {str(v): {v} for v in unique_values}
                    
            # Handle numeric type
            if pd.api.types.is_integer_dtype(self.df[key]) or pd.api.types.is_float_dtype(self.df[key]):
                print(f"key:{key} \nvalues:{self.df[key].dtype}")
                print(f"example: {self.df[key].head().tolist()}")

                unique_vals = self.df[key].dropna().unique()
                num_unique = len(unique_vals)
                val_range = unique_vals.max() - unique_vals.min() if num_unique > 0 else 0
                
                # 对于小范围离散值（如评分1-5），直接将所有值放到一个组
                if num_unique <= 10 and val_range <= 10:
                    print(f"  -> Small discrete range detected ({num_unique} unique values, range={val_range}), grouping all together")
                    all_vals = sorted(unique_vals.tolist())
                    partitions[key] = [all_vals]
                else:
                    # 原有的delta分组逻辑
                    for value in self.df[key].unique():
                        assigned = False
                        
                        for partion in partitions[key]:
                            
                            if abs(value - partion[0]) <= partion[0] * self.delta:
                                partion.append(value)
                                assigned = True
                                break
                        
                        if not assigned:
                            partitions[key].append([value])
                    
                    for partition in partitions[key]:
                        partition.sort()
                    
                    column_groups = []
                    for partition in partitions[key]:
                        if len(partition) > 1:
                            column_groups.extend(self.dp_grouping(partition))
                        else:
                            column_groups.append(partition)  # 单值分区直接添加
                    
                    partitions[key] = column_groups
            
        return partitions
            

    @staticmethod
    def Hash(key, index, extra_param=None):
        if extra_param is not None:
            combined = f"{key}{index}{extra_param}"
        else:
            combined = f"{key}{index}"
        num = int(hashlib.sha256(combined.encode('utf-8')).hexdigest(), 16)
        return num % (2**32)
    
    # def generate_seed(key, index):
    #     random.seed(DomainProcessor.Hash(key, index))
    #     random_values = [int(random.random() * 1e16) for _ in range(3)]
    #     return random_values
    
    @staticmethod
    def generate_seed(key, index, extra_param=None):
        if extra_param is not None:
            random.seed(DomainProcessor.Hash(key, index, extra_param))
        else:
            random.seed(DomainProcessor.Hash(key, index))
        random_values = [int(random.random() * 1e16) for _ in range(3)]
        return random_values


    @staticmethod        
    def find_closest_value(v, domain_group):
        if domain_group is None:
            return None
            
        if isinstance(v, (int, float, np.integer, np.floating)):
            # 如果domain_group是列表类型（数值数据的情况）
            if isinstance(domain_group, list):
                for sublist in domain_group:
                    if isinstance(sublist, list) and v in sublist:
                        return sublist
            return None
        else:
            # 如果domain_group是字典类型（类别数据的情况）
            if isinstance(domain_group, dict):
                for key, value in domain_group.items():
                    value_set = value if isinstance(value, set) else set(value)
                    value_set.add(key)
                    if v in value_set:
                        return value_set
            return None

def convert_int64_to_int(data):
    new_data = {}
    for key, value in data.items():
        new_value = {int(inner_key): inner_value for inner_key, inner_value in value.items()}
        new_data[key] = new_value
    return new_data

def convert_sets_to_lists(data):
    for key, value in data.items():
        for inner_key, inner_value in value.items():
            value[inner_key] = list(inner_value)
    return data

def convert_lists_to_sets(data):
    for key, value in data.items():
        for inner_key, inner_value in value.items():
            value[inner_key] = set(inner_value)
    return data