def parseCode(data_str):
    # Initialize an empty dictionary to store key-value pairs
    data = {}
    
    # Remove outer curly braces from the input string
    inner_data = data_str.strip('{}')
    
    # Split the inner data string by commas to separate key-value pairs
    pairs = inner_data.split(',')
    
    # Iterate over each key-value pair
    for pair in pairs:
        # Split the pair by colon to separate key and value
        key, value = pair.split(':')
        
        # Strip double quotes from key and value strings
        key = key.strip().strip('"')
        value = value.strip().strip('"')
        
        # Add the key-value pair to the data dictionary
        data[key] = value
    
    return data

if __name__ == "__main__":
    data_str = '{"name": "John Doe", "age": "30", "city": "New York"}'
    result = parseCode(data_str)
    print(result)