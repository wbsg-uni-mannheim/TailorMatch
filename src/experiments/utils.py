def clean_response(response):
    if "yes" in response.lower():
        return 1
    elif "no" in response.lower():
        return 0
    else:
        return -1
    
# Function to insert product descriptions into the prompt
def insert_product_descriptions(prompt_template: str, product1: str, product2: str):
    # Replace placeholder texts with actual product descriptions
    prompt = prompt_template.replace("'Entity 1'", product1).replace("'Entity 2'", product2)
    return prompt

def generate_question(prompt, entity1, entity2):
    prompt = insert_product_descriptions(prompt, entity1, entity2)
    return [
        {"role": "user", "content": prompt},
    ]
    
    
