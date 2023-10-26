import os
import pandas as pd
from datasets import load_dataset
import openai
import pinecone
import time
from tqdm.auto import tqdm
from nemoguardrails import LLMRails, RailsConfig
from dotenv import load_dotenv

# Initialize environment variables
load_dotenv()
embed_model_id = "text-embedding-ada-002"  # Replace with your model ID
api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
env = os.getenv("PINECONE_ENVIRONMENT") or "YOUR_ENV"

data = load_dataset(
    "medicine/Medicine_Details.csv",
    split="train"
)
data

# Initialize Pinecone
pinecone.init(api_key=api_key, environment=env)
index_name = "nemo-guardrails-rag-with-actions"

pinecone.whoami()

if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine'
    )
    # wait for index to be initialized
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pinecone.Index(index_name)
# view index stats
index.describe_index_stats()

# Connect to index
index = pinecone.Index(index_name)

print(index_name)

# Read CSV in chunks
chunk_size = 100  # Adjust based on your memory capacity
reader = pd.read_csv("medicine/Medicine_Details.csv", chunksize=chunk_size)

for i in tqdm(range(0, len(data), batch_size)):
    # find end of batch
    i_end = min(len(data), i+batch_size)
    batch = data[i:i_end]
    # get ids
    ids_batch = batch['uid'].to_list()
    # get texts to encode
    texts = batch['chunk'].to_list()
    # create embeddings
    res = openai.Embedding.create(input=texts, engine=embed_model_id)
    embeds = [record['embedding'] for record in res['data']]
    # create metadata
    metadata = [{
        'chunk': x['chunk'],
        'source': x['source']
    } for _, x in batch.iterrows()]
    to_upsert = list(zip(ids_batch, embeds, metadata))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)        



    # rag functions for guardrails
    
async def retrieve(query: str) -> list:
    # create query embedding
    res = openai.Embedding.create(input=[query], engine=embed_model_id)
    xq = res['data'][0]['embedding']
    # get relevant contexts from pinecone
    res = index.query(xq, top_k=5, include_metadata=True)
    # get list of retrieved texts
    contexts = [x['metadata']['name'] for x in res['matches']]
    return contexts

async def rag(query: str, contexts: list) -> str:
    print("> RAG Called")  # we'll add this so we can see when this is being used
    context_str = "\n".join(contexts)
    # place query and contexts into RAG prompt
    prompt = f"""You are a helpful assistant, below is a query from a user and
    some relevant contexts. Answer the question given the information in those
    contexts. If you cannot find the answer to the question, say "I don't know".

    Contexts:
    {context_str}

    Query: {query}

    Answer: """
    # generate answer
    res = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens=100
    )
    return res['choices'][0]['text']

#initialize configs for guardrails

yaml_content = """
models:
- type: main
  engine: openai
  model: text-davinci-003
"""

rag_colang_content = """
# define limits
# Define invalid questions
define user ask politics
    "what are your political beliefs?"
    "thoughts on the president?"
    "left wing"
    "right wing"

define bot answer politics
    "I'm an AI assistant focused on medicine, I cannot answer political questions."
    "My role is to answer medicine-related questions using trusted data sources."  

define flow politics
    user ask politics  
    bot answer politics
    bot offer help

# Define medicine info questions 
define user ask medicine
    "what are the side effects of ibuprofen?"
    "is there a substitute for penicillin?"
    "what class of medicine is metformin?"
    "how is aspirin used?"

# Define medicine Q&A flow
define flow medicine
    user ask medicine
    $contexts = execute retrieve(query=$last_user_message)  
    $answer = execute rag(query=$last_user_message, contexts=$contexts)
    bot $answer
"""



# initialize rails config
config = RailsConfig.from_content(
    colang_content=rag_colang_content,
    yaml_content=yaml_content
)
# create rails
rag_rails = LLMRails(config)

rag_rails.register_action(action=retrieve, name="retrieve")
rag_rails.register_action(action=rag, name="rag")

#testing rag

await rag_rails.generate_async(prompt="hello")

await rag_rails.generate_async(prompt="what is a use for amlopin 5mg?")

no_rag_colang_content = """
# define limits
define user ask politics
    "what are your political beliefs?"
    "thoughts on the president?"
    "left wing"
    "right wing"

define bot answer politics
    "I'm a shopping assistant, I don't like to talk of politics."
    "Sorry I can't talk about politics!"

define flow politics
    user ask politics
    bot answer politics
    bot offer help
"""

# initialize rails config
config = RailsConfig.from_content(
    colang_content=no_rag_colang_content,
    yaml_content=yaml_content
)
# create rails
no_rag_rails = LLMRails(config)

# with RAG
await rag_rails.generate_async(
    prompt="what are some uses for amlopin 5mg?"
)