import os
import pandas as pd
import openai
import pinecone
import time
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Initialize environment variables
load_dotenv()
embed_model_id = "text-embedding-ada-002"  # Replace with your model ID
api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
env = os.getenv("PINECONE_ENVIRONMENT") or "YOUR_ENV"

# Initialize Pinecone
pinecone.init(api_key=api_key, environment=env)
index_name = "nemo-guardrails-rag-with-actions"

# Create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    # Create index (dimension needs to be known beforehand)
    pinecone.create_index(index_name, dimension=512, metric='cosine')
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to index
index = pinecone.Index(index_name)

# Read CSV in chunks
chunk_size = 500  # Adjust based on your memory capacity
reader = pd.read_csv("medicine/Medicine_Details.csv", chunksize=chunk_size)

# Loop to read and process each chunk
for i, chunk in enumerate(reader):
    # Data preprocessing
    data = chunk[['Medicine Name', 'Composition', 'Uses', 'Side_effects', 'Image URL', 'Manufacturer', 'Excellent Review %', 'Average Review %', 'Poor Review %']]
    data['uid'] = data['Medicine Name'].astype(str) + '-' + str(i)
    
    # Embedding and upserting
    ids_batch = data['uid'].tolist()
    texts = data['Medicine Name'].tolist()
    res = openai.Embedding.create(input=texts, engine=embed_model_id)
    embeds = [record['embedding'] for record in res['data']]
    
    # Create metadata
    metadata = [{
        'Medicine Name': row['Medicine Name'],
        'Composition': row['Composition'],
        'Uses': row['Uses'],
        'Side_effects': row['Side_effects'],
        'Manufacturer': row['Manufacturer'],
    } for _, x in data.iterrows()]
    
    to_upsert = list(zip(ids_batch, embeds, metadata))
    index.upsert(vectors=to_upsert)

for idx, row in data.iterrows():
    print(f"Processing row {idx}")
    # existing code
with open("debug_log.txt", "a") as f:
    f.write(f"Processing row {idx}\n")
