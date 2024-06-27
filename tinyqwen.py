from langchain_core.prompts import PromptTemplate
import re
import traceback
import sys
import readline
from langchain_huggingface import HuggingFacePipeline

# this model needs GPU and CUDA.
#
# pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
# pip install auto-gptq optimum

max_new_tokens=300
task="text-generation"
model="Qwen/Qwen-1_8B-Chat-Int4"
files = []

hf = HuggingFacePipeline.from_model_id(
    model_id=model,
    task=task,
    trust_remote_code=True,
    pipeline_kwargs={
        "max_new_tokens": max_new_tokens,
    },
    model_kwargs={
        "trust_remote_code":True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True
    }
)

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.runnables import RunnablePassthrough

# load the document and split it into chunks
documents = []
for f in files:
    documents.extend(TextLoader(f).load())

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embed = SentenceTransformerEmbeddings(model_name=model)

# load it into Chroma
vec = Chroma.from_documents(docs, embed)
retri = vec.as_retriever() # default: k is 4, num of doc

def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)

template = """你是一个负责解答疑问的助手，请根据以下上下文信息来回答问题。如果你不知道问题的答案，请你回答不知道。你的回答要尽可能严谨简洁。

问题: {question}

上下文: {context}

答案:"""

prompt = PromptTemplate.from_template(template)

chain = (
    {"context": retri | format_docs, "question": RunnablePassthrough()}
    | prompt
    | hf.bind()
)

print("\n\n")
ans_pat = "^.*答案: *(.*)$"
while True:
    try:
        print("Enter your question:")
        q = None
        while not q: q = sys.stdin.readline().strip()
        print()

        resp = chain.invoke(q)
        m = re.search(ans_pat, resp, re.DOTALL)
        ans = resp
        if m: ans = m[1]

        # print(resp)
        # print(f"\n\n>>>> resp: '{resp}'")
        print(f"\n> {ans}\n")

    except InterruptedError:
        sys.exit()
    except Exception as e:
        print("Exception caught", e)
        traceback.print_exc()
