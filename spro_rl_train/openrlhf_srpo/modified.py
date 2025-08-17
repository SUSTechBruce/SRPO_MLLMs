import json
import os

def modify_image_paths(input_file, output_file, new_path="/mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/Dataset_and_models/images"):
    # Open input and output files
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        # Process each line (JSON object)
        for line in f_in:
            # Parse the JSON line
            entry = json.loads(line.strip())
            
            # Parse the message JSON string
            message = json.loads(entry['message'])
            
            # Access the content array
            content = message[0]['content']
            
            # Find the image content and modify the path
            for item in content:
                if item['type'] == 'image':
                    # Get the filename from the original path
                    filename = os.path.basename(item['image'])
                    # Create new path
                    item['image'] = os.path.join(new_path, filename)
            
            # Update the message in the entry
            entry['message'] = json.dumps(message)
            
            # Write the modified entry to the output file as a JSON line
            json.dump(entry, f_out)
            f_out.write('\n')

# Example usage
if __name__ == "__main__":
    input_file = "/mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/Dataset_and_models/39Krelease.jsonl"  # Input JSONL file
    output_file = "/mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/Dataset_and_models/modified_39Krelease.jsonl"  # Output JSONL file
    modify_image_paths(input_file, output_file)