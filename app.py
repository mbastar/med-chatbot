import os
import openai
import pinecone
import time
from tqdm.auto import tqdm
from nemoguardrails import LLMRails, RailsConfig
from dotenv import load_dotenv

from datasets import load_dataset

data = load_dataset("medicine_dataset", split="train")
data

data = data.map(lambda x: {
    'uid': f"{x['doi']}-{x['chunk-id']}"
})
data

data = data.to_pandas()
# drop irrelevant fields
data = data[['uid', 'chunk', 'title', 'source']]



embed_model_id = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "We would have some text to embed here",
        "And maybe another chunk here too"
    ], engine=embed_model_id
)

pinecone.init(      
	api_key='10b963ce-a1ac-4237-a3e4-c0c3a7795e41',      
	environment='gcp-starter'      
)      
index = pinecone.Index('nemo-guardrails-rag-with-actions')

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") or "YOUR_API_KEY"
# find your environment next to the api key in pinecone console
env = os.getenv("PINECONE_ENVIRONMENT") or "YOUR_ENV"

pinecone.init(api_key=api_key, environment=env)
pinecone.whoami()

index_name = "nemo-guardrails-rag-with-actions"



# check if index already exists (it shouldn't if this is first time)
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



batch_size = 100  # how many embeddings we create and insert at once

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
    contexts = [x['metadata']['chunk'] for x in res['matches']]
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

# define RAG intents and flow
define user ask llama
    "tell me about llama 2?"
    "what is large language model"
    "where did meta's new model come from?"
    "how to llama?"
    "have you ever meta llama?"

define flow llama
    user ask llama
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

await rag_rails.generate_async(prompt="tell me about llama 2")

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
    prompt="what was red teaming used for in llama 2 training?"
)