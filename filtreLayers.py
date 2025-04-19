import re
import os
from datetime import datetime

def extract_unique_base_names(filepath="full.txt"):
    """
    Reads a file containing LoRA keys, removes specified suffixes,
    and returns a sorted list of unique base names.

    Args:
        filepath (str): The path to the input text file.

    Returns:
        list: A sorted list of unique base key names, or an empty list on error.
    """
    # Suffixes to remove. Sorted by length descending to handle longer matches first
    suffixes_to_remove = sorted([
        ".alpha",
        ".dora_scale",
        ".lora_down.weight",
        ".lora_up.weight"
    ], key=len, reverse=True)

    unique_base_names = set()

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Basic check to process only lines that likely contain keys
                if not line or line.startswith("===") or line.startswith("("):
                    continue

                original_line = line
                processed = False
                for suffix in suffixes_to_remove:
                    if line.endswith(suffix):
                        # Remove the suffix to get the base name
                        base_name = line[:-len(suffix)]
                        unique_base_names.add(base_name)
                        processed = True
                        break # Stop checking suffixes once one is found and removed

                # Optional: Handle lines that might be keys but don't end with a known suffix
                # (In this specific input, it seems all keys do have these suffixes.)
                # if not processed and '.' in line and 'lora_' in line:
                #     print(f"Warning: Line '{original_line}' processed as potentially unique base name (no known suffix found).")
                #     unique_base_names.add(original_line)


    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred during reading/processing: {e}")
        return []

    # Return the unique names as a sorted list for consistent output
    return sorted(list(unique_base_names))

def save_list_to_file(data_list, filename):
    """
    Saves a list of strings to a text file, one item per line.

    Args:
        data_list (list): The list of strings to save.
        filename (str): The desired output filename.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as outfile:
            for item in data_list:
                outfile.write(item + "\n")
        return True
    except Exception as e:
        print(f"An error occurred while writing to file {filename}: {e}")
        return False

# --- Main execution ---
if __name__ == "__main__":
    input_filename = "full.txt"
    unique_keys = extract_unique_base_names(input_filename)

    if unique_keys:
        # Generate timestamp
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S") # Format: YYYYMMDD_HHMMSS

        # Construct output filename
        output_filename = f"unique_lora_keys_{timestamp}.txt"

        print(f"Found {len(unique_keys)} unique base LoRA keys.")

        # Save the list to the timestamped file
        if save_list_to_file(unique_keys, output_filename):
            print(f"Output successfully saved to: {output_filename}")
        else:
            print("Failed to save the output file.")

    else:
        # Handle case where no keys were found or an error occurred in extraction
        print("No unique keys were extracted. No output file generated.")