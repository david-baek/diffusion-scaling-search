# https://github.com/arthw/imagenet_label/blob/master/imagenet_1000.txt

# Specify the paths for the input and output files.
input_file_path = 'imagenet_1000.txt'
output_file_path = 'prompts_imagenet.txt'

with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        # Split the line into at most two parts (index and prompt)
        parts = line.split(' ', 1)  # Split on the first space to separate index and prompt
        assert len(parts) == 2, f"Line does not contain exactly two parts: {line.strip()}"
        # Write only the prompt (without the index)
        outfile.write(parts[1])