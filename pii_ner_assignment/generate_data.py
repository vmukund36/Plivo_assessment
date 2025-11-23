import json
import random
import faker
from tqdm import tqdm
import os

fake = faker.Faker()

def get_noisy_text_and_labels():
    # 1. Templates for sentence structures
    templates = [
        "my {label} is {entity}",
        "the {label} is {entity}",
        "please record {entity} as the {label}",
        "it is {entity}",
        "{entity} is the {label}",
        "i can be reached at {entity}",
        "contact me at {entity}",
        "payment with {entity}",
        "born on {entity}",
        "visiting {entity}",
    ]

    # 2. Select an entity type to generate
    choices = ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION", "O"]
    label_type = random.choice(choices)
    
    entity_text = ""
    prefix_text = ""
    
    # 3. Generate Entity Data with STT Noise
    if label_type == "CREDIT_CARD":
        # STT often spaces out digits or says "credit card"
        entity_text = fake.credit_card_number()
        # Noise: replace dashes with spaces, sometimes spell out numbers (simplified here to spaces)
        entity_text = entity_text.replace("-", " ") 
        prefix_text = random.choice(["credit card", "card number", "number"])
        
    elif label_type == "PHONE":
        entity_text = fake.phone_number()
        # Noise: remove parentheses, replace dashes
        entity_text = entity_text.replace("(", "").replace(")", "").replace("-", " ").replace(".", " ")
        prefix_text = random.choice(["phone", "cell", "mobile", "number"])

    elif label_type == "EMAIL":
        entity_text = fake.email()
        # Crucial STT Noise: "dot" and "at"
        entity_text = entity_text.replace("@", " at ").replace(".", " dot ")
        prefix_text = random.choice(["email", "address", "mail"])

    elif label_type == "PERSON_NAME":
        entity_text = fake.name()
        prefix_text = random.choice(["name", "person", "client"])

    elif label_type == "DATE":
        entity_text = fake.date(pattern="%d %B %Y") # e.g. 12 January 2022
        prefix_text = random.choice(["date", "birthday", "dob"])

    elif label_type == "CITY":
        entity_text = fake.city()
        prefix_text = random.choice(["city", "town"])

    elif label_type == "LOCATION":
        entity_text = fake.street_address()
        prefix_text = random.choice(["location", "address", "place"])

    elif label_type == "O":
        # Generate generic filler text
        entity_text = fake.sentence(nb_words=6).replace(".", "")
        prefix_text = ""
    
    # 4. Construct Sentence
    if label_type == "O":
        full_text = entity_text.lower()
        return full_text, []
    
    # Pick a template and fill it
    template = random.choice(templates)
    # Fallback for simple concatenation if template fails logic
    full_text = f"{prefix_text} {entity_text}"
    
    # Construct text with simple logic to track indices
    pre_filler = random.choice(["hi ", "hello ", "ok ", "so ", ""])
    connector = random.choice([" is ", " equals ", " "])
    
    # Assembly: "pre_filler" + "prefix" + "connector" + "entity"
    # We must lowercase everything to match uncased models
    part1 = (pre_filler + prefix_text + connector).lower()
    part2 = entity_text.lower()
    
    full_text = part1 + part2
    
    start_idx = len(part1)
    end_idx = start_idx + len(part2)
    
    return full_text, [{"start": start_idx, "end": end_idx, "label": label_type}]

def generate_file(filename, count):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for i in tqdm(range(count), desc=f"Generating {filename}"):
            text, entities = get_noisy_text_and_labels()
            obj = {
                "id": f"{i}",
                "text": text,
                "entities": entities
            }
            f.write(json.dumps(obj) + "\n")

if __name__ == "__main__":
    # Generate 1000 train, 200 dev
    generate_file("data/train.jsonl", 1000)
    generate_file("data/dev.jsonl", 200)
    print("Data generation complete.")