import re
from datasets import load_dataset

# Load the DrawBench dataset (using the default split)
dataset = load_dataset("shunk031/DrawBench")

prompt_field = "prompts"

# Open an output file to write the cleaned prompts.
output_file_path = "../prompts_drawbench.txt"
with open(output_file_path, "w", encoding="utf-8") as outfile:
    # Iterate over the test split (or change to the desired split)
    for row in dataset["test"]:
        # Get the prompt text from the row
        prompt_text = row[prompt_field]
        outfile.write(prompt_text + "\n")

print(f"Drawbench prompts have been written to {output_file_path}")
