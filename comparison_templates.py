Direct_Template = """
Write a Python function that converts the given data snippet into a JSON format. Output only the Python code required for this conversion, without any additional explanations.
data: {data}
"""

COT_Template = """
Write a Python function that converts the given data snippet into a JSON format. Output only the Python code required for this conversion, without any additional explanations. \
data: {data}
First, consider the process step by step, and then present the Python script:
"""

Few_shot_Template = """
You are a data format conversion expert. Given a random data snippet that represents a complete data object from the original dataset, \
you need to generate Python code that converts the given data snippet into JSON format. Ensure that the original dataset's data objects can also be correctly converted to JSON using the same Python code, \
without modifying the structure of the given data snippet.
"""

example1 = [
    {
        "snippet": """<Person Age="26" Gender="Female" Income="&lt;=50K"><Work><Workclass>Federal-gov</Workclass><Occupation>Exec-managerial</Occupation><HoursPerWeek>40</HoursPerWeek></Work><Education><Level>Bachelors</Level><EducationNum>13</EducationNum></Education><Financial><FinalWeight>48099</FinalWeight><CapitalGain>0</CapitalGain><CapitalLoss>0</CapitalLoss></Financial><Personal><MaritalStatus>Never-married</MaritalStatus><Relationship>Not-in-family</Relationship><Race>White</Race><NativeCountry>United-States</NativeCountry></Personal></Person>""",
        "answer": """
import json
from xml.etree import ElementTree as ET

def convert_to_json(data: str) -> dict:
    def parse_element(element):
        parsed_dict = {{}}

        # Process attributes
        for key, value in element.attrib.items():
            parsed_dict[f"@{{key}}"] = value

        # Process child elements
        children = list(element)
        if children:
            # Nested dictionary for child elements
            child_dict = {{}}
            for child in children:
                child_result = parse_element(child)
                if child.tag not in child_dict:
                    child_dict[child.tag] = child_result
                else:
                    # If the key already exists, make it a list
                    if not isinstance(child_dict[child.tag], list):
                        child_dict[child.tag] = [child_dict[child.tag]]
                    child_dict[child.tag].append(child_result)
            parsed_dict.update(child_dict)
        else:
            # If no children, use the text content
            if element.text and element.text.strip():
                parsed_dict[element.tag] = element.text.strip()

        return parsed_dict

    try:
        # Parse the XML-like data
        root = ET.fromstring(data)
        # Convert the XML tree to a JSON dictionary
        json_dict = {{root.tag: parse_element(root)}}
        return json_dict
    except ET.ParseError as e:
        # Handle XML parsing errors
        print(f"XML parsing error: {{e}}")
        return {{}}
""",
    }]


example2 = [
    {
        "snippet": """timestamp~public_transport_usage~traffic_flow~bike_sharing_usage~pedestrian_count~weather_conditions~day_of_week~holiday~event~temperature~humidity~road_incidents~public_transport_delay~bike_availability~pedestrian_incidents\n2099-06-27 23:00:00~491~4368~153~1946~Fog~Saturday~0~~5.999130786079283~49~8~26.261010378186725~65~3""",
        "answer": """
import json

def convert_to_json(data_str):
    # Split the input string into lines and ensure there are at least two lines for headers and values.
    lines = data_str.strip().split('\n')
    if len(lines) < 2:
        raise ValueError("Input data is not in the expected format.")

    # Split the first line to get the headers/tags and the second line to get the corresponding values.
    headers = lines[0].split('~')
    values = lines[1].split('~')
    
    # Initialize an empty dictionary to store the parsed data.
    data_dict = {{}}
    
    # Iterate over each header-value pair
    for tag, value in zip(headers, values):
        # Convert empty strings to None
        if value == '':
            data_dict[tag] = None
        else:
            # Convert values to integers or floats where applicable
            try:
                # Try converting the value to an integer
                data_dict[tag] = int(value)
            except ValueError:
                try:
                    # If integer conversion fails, try converting to float
                    data_dict[tag] = float(value)
                except ValueError:
                    # If both conversions fail, keep it as a string
                    data_dict[tag] = value

    # Nest the dictionary under the key 'data'
    json_dict = {'data': data_dict}
    
    # Convert the dictionary to a JSON-formatted string with indentation for readability
    return json.dumps(json_dict, indent=4)
""",
    }]


examples = [
    {
        "snippet": """<Person Age="26" Gender="Female" Income="&lt;=50K"><Work><Workclass>Federal-gov</Workclass><Occupation>Exec-managerial</Occupation><HoursPerWeek>40</HoursPerWeek></Work><Education><Level>Bachelors</Level><EducationNum>13</EducationNum></Education><Financial><FinalWeight>48099</FinalWeight><CapitalGain>0</CapitalGain><CapitalLoss>0</CapitalLoss></Financial><Personal><MaritalStatus>Never-married</MaritalStatus><Relationship>Not-in-family</Relationship><Race>White</Race><NativeCountry>United-States</NativeCountry></Personal></Person>""",
        "answer": """
import json
from xml.etree import ElementTree as ET

def convert_to_json(data: str) -> dict:
    def parse_element(element):
        parsed_dict = {{}}

        # Process attributes
        for key, value in element.attrib.items():
            parsed_dict[f"@{{key}}"] = value

        # Process child elements
        children = list(element)
        if children:
            # Nested dictionary for child elements
            child_dict = {{}}
            for child in children:
                child_result = parse_element(child)
                if child.tag not in child_dict:
                    child_dict[child.tag] = child_result
                else:
                    # If the key already exists, make it a list
                    if not isinstance(child_dict[child.tag], list):
                        child_dict[child.tag] = [child_dict[child.tag]]
                    child_dict[child.tag].append(child_result)
            parsed_dict.update(child_dict)
        else:
            # If no children, use the text content
            if element.text and element.text.strip():
                parsed_dict[element.tag] = element.text.strip()

        return parsed_dict

    try:
        # Parse the XML-like data
        root = ET.fromstring(data)
        # Convert the XML tree to a JSON dictionary
        json_dict = {{root.tag: parse_element(root)}}
        return json_dict
    except ET.ParseError as e:
        # Handle XML parsing errors
        print(f"XML parsing error: {{e}}")
        return {{}}
""",
    },
    {
        "snippet": """timestamp~public_transport_usage~traffic_flow~bike_sharing_usage~pedestrian_count~weather_conditions~day_of_week~holiday~event~temperature~humidity~road_incidents~public_transport_delay~bike_availability~pedestrian_incidents\n2099-06-27 23:00:00~491~4368~153~1946~Fog~Saturday~0~~5.999130786079283~49~8~26.261010378186725~65~3""",
        "answer": """
import json

def convert_to_json(data_str):
    # Split the input string into lines and ensure there are at least two lines for headers and values.
    lines = data_str.strip().split('\n')
    if len(lines) < 2:
        raise ValueError("Input data is not in the expected format.")

    # Split the first line to get the headers/tags and the second line to get the corresponding values.
    headers = lines[0].split('~')
    values = lines[1].split('~')
    
    # Initialize an empty dictionary to store the parsed data.
    data_dict = {{}}
    
    # Iterate over each header-value pair
    for tag, value in zip(headers, values):
        # Convert empty strings to None
        if value == '':
            data_dict[tag] = None
        else:
            # Convert values to integers or floats where applicable
            try:
                # Try converting the value to an integer
                data_dict[tag] = int(value)
            except ValueError:
                try:
                    # If integer conversion fails, try converting to float
                    data_dict[tag] = float(value)
                except ValueError:
                    # If both conversions fail, keep it as a string
                    data_dict[tag] = value

    # Nest the dictionary under the key 'data'
    json_dict = {'data': data_dict}
    
    # Convert the dictionary to a JSON-formatted string with indentation for readability
    return json.dumps(json_dict, indent=4)
""",
    }
]

