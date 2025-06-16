import json

def parse_explanations(explanations: list[str]) -> list[dict]:
    error_count = 0
    records = []
    for explanation in explanations:
        record = []
        lines = explanation.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|||")
            parsed = {}
            for p in parts:
                key_val = p.split("=", 1)
                if len(key_val) == 2:
                    # Store value; strip whitespace. Keep empty strings for now.
                    parsed[key_val[0].strip()] = key_val[1].strip()
                elif len(key_val) == 1 and key_val[0]: # Handle keys potentially without values like "attribute="
                    parsed[key_val[0].strip()] = "" # Store as empty string if no value after '='

            # --- Start processing parsed parts, defaulting to None ---
            attribute = parsed.get("attribute") # Get value or None if key missing
            if not attribute: # Skip if attribute name is missing or empty
                # print(f"Skipping line due to missing attribute name: {line}") # Optional debug info
                continue

            importance = None # Default importance to None
            importance_str = parsed.get("importance", None)
            if importance_str is not None: # Check if importance key exists
                try:
                    importance = float(importance_str)
                except ValueError:
                    print(f"Error parsing importance '{importance_str}' in line: {line}. Using None.")
                    error_count += 1
                    # Keep importance as None

            similarity = None # Default similarity to None
            similarity_str = parsed.get("similarity", None)
            if similarity_str is not None: # Check if similarity key exists
                if similarity_str.lower() == "missing":
                    similarity = None # Explicitly None if "missing"
                else:
                    try:
                        similarity = float(similarity_str)
                    except ValueError:
                        print(f"Error parsing similarity '{similarity_str}' in line: {line}. Using None.")
                        error_count += 1
                        # Keep similarity as None

            # Process values, aiming for None if a part is missing/empty
            value1 = None
            value2 = None
            values_str = parsed.get("values") # Get value string or None

            if values_str is not None: # Only proceed if 'values=' part existed
                vals = values_str.split("###")
                # Assign value1 if the first part exists and is not empty
                if len(vals) >= 1 and vals[0]:
                    value1 = vals[0]
                # Assign value2 if the second part exists and is not empty
                if len(vals) == 2 and vals[1]:
                     value2 = vals[1]
                # Note: If values_str was "" (e.g., from values=), vals will be [''],
                # value1 and value2 will correctly remain None.
                # If values_str was "###" (e.g., from values=###), vals will be ['', ''],
                # value1 and value2 will correctly remain None.

            # Skip record only if *both* values ended up as None AND the original values string
            # was effectively empty or just the separator (this avoids skipping attribute=X|||values=Y)

            record.append({
                "attribute": attribute,       
                "importance": importance,     
                "similarity": similarity,     
                "value1": value1,            
                "value2": value2             
            })

        # Check if the record for this explanation is empty after processing all lines
        if record: # Only append if we actually parsed something for this explanation
             records.append(record)
        elif explanation.strip(): # If the original explanation wasn't empty but we parsed nothing
            print(f"Warning: No valid attribute lines found in explanation:\n{explanation}")

    if error_count > 0:
        print(f"Total parsing errors encountered: {error_count}")
    return records

def df_add_explanations(file_path, df):
   """
   Add explanations and parsed explanations from a JSON file to a dataframe
   
   Args:
       file_path (str): Path to JSON file containing explanations
       df (pd.DataFrame): DataFrame to add explanations to
       
   Returns:
       pd.DataFrame: DataFrame with added explanation columns
   """
   with open(file_path, "r") as f:
       data = json.load(f)
       
   explanations = [x["explanation"] for x in data["examples"]]
   parsed_explanations = parse_explanations(explanations)
   
   df["explanations"] = explanations
   df["parsed_explanations"] = parsed_explanations
   
   return df