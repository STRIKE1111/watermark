Separators_template = """
Identify and list the most possible separators of data. Explain within 30 words. \
{data}
"""

Nest_template = """
Do you think the structure of data is nested? Do not assume anything not explicitly mentioned. Explain within 50 words. \
{data}
"""

Tag_template = """
Do you think there's tags or labels in the data? Explain within 100 words. \
{data}
"""

Result_example_template = """
Understand separators, nest, type and other components of the data. convert the following data into a JSON dictionary. \
For example, use {{\"key1\": \"value1\", \"key2\": \"value2\", ...}} to represent a dictionary. No explanation. Only output the final json result.

Input data: {data}
separators:{separators}
nest:{nest}
tags:{tags}
"""

Problem_template = """
The auxiliary information are separators, nest, tags. \
separators: {separators}
nest: {nest}
tags: {tags}

Converts the given data snippet into JSON format. \
Input data: {data}.
Desired json data: {json_data}.
"""

Function_gen_template = """
Please understand the requirement and write a rough solving process. \
It starts with an input-output structure. Input is a string containing the data to be converted. Output is a dictionary representing the converted data in JSON format. \
You should use three basic structures to build the solving process, including sequences, branches, and loops. The necessary details should be written in natural languages. \
The function name is *parseCode*. The function takes a single string argument ’data_str’ and returns the ’data’ dictionary containing all parsed key-value pairs. \
The necessary details are written in natural languages. \
requirement: {requirement}
"""

Code_gen_template = """
Solving process: {solving_process}
Please check the above solving process and write a python code based on it. \
Note that the solving process may contain errors. \
To clarify the logic, it’s preferable to include annotations in the script. \
If there exist tags, ensure that all tags are converted with a special prefix(e.g., ’@’). \
Ensure the generated code can cover all instances. \
All test code is placed within the if __name__ == "__main__": block so that it only executes when the file is run directly and not when imported by other files. \
Here’s the Python code for the given issue:
"""

Python_code_template = """Extract only the code and its comments from the generated content that has an identifier of ```python```, \
Don't output it as a code block. \
so you can write them later in a python file. Don't output any results that are not relevant to the requirements. \
final_python_code_result: {parseCode}
"""

