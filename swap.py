import json
import os

def copy_and_replace(dataset_file, json_file, output_folder):
    with open(json_file, 'r') as jf:
        replace_dict = json.load(jf)

    base_filename = os.path.basename(dataset_file)

    with open(dataset_file, 'r') as df:
        lines = df.readlines()

    for line_num, (match_value, replace_value) in replace_dict.items():
        line_num = int(line_num)  
        if line_num <= len(lines):  
            line = lines[line_num]  

            line_str = str(line)
            
            if isinstance(match_value, (int, float)):
                match_value = str(match_value)

            if match_value in line_str:
                line = line.replace(match_value, str(replace_value))  # 替换为字符串

            lines[line_num] = line

    
    output_file = os.path.join(output_folder, f"watermarked_{base_filename}")
    with open(output_file, 'w') as of:
        of.writelines(lines)
    
    print(f"The modified file has been saved to: {output_file}")

dataset_file = 'parser_data\code1\dataset\logfiles.log'  
json_file = 'parser_data\code1\swap\swap.json'  
output_folder = 'parser_data\code1\watermarked_dataset'  

os.makedirs(output_folder, exist_ok=True)

copy_and_replace(dataset_file, json_file, output_folder)
