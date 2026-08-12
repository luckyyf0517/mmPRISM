import json

# Read original JSON file
input_file = "dataset/CSL_News_Labels.json"
output_file = "dataset/CSL_News_Labels_converted.json"

# Convert data
def convert_json_format():
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('data length: ', len(data))
    
    # Create dictionary in new format
    converted_data = {}
    for item in data:
        # Get seq_id from video filename (remove extension)
        seq_id = item['video'].split('.')[0]
        converted_data[seq_id] = item['text']
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    convert_json_format()
    print(f"Conversion complete! Results saved to {output_file}")