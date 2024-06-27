from langchain_core.prompts import PromptTemplate
import re
import traceback
import sys
import readline
from langchain_huggingface import HuggingFacePipeline

max_new_tokens=300
task="text-generation"
model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

hf = HuggingFacePipeline.from_model_id(
    model_id=model,
    task=task,
    pipeline_kwargs={
        "max_new_tokens": max_new_tokens,
    },
    model_kwargs={
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
    },
)

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

# load the document and split it into chunks
# e.g.,
#
# onecafe is a company found by yongjie.zhuang.
#
# onecafe sells coffee.
files = ["about.txt"]
documents = []
for f in files:
    documents.extend(TextLoader(f).load())

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(">> docs", docs)

# create the open-source embedding function
embed = HuggingFaceEmbeddings()

# load it into Chroma
vec = Chroma.from_documents(docs, embed)
reti = vec.as_retriever(search_kwargs={"k": 10}) # default: k is 4

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
ans_pat = "^.*Answer: *(.*)$"
prompt = PromptTemplate.from_template(template)

chain = (
    {"context": reti | format_docs, "question": RunnablePassthrough()}
    | prompt
    | hf.bind()
)

print("\n\n")

while True:
    try:
        print("Enter your question:")
        q = None
        while not q: q = sys.stdin.readline().strip()

        resp = chain.invoke(q)
        m = re.search(ans_pat, resp, re.DOTALL)
        ans = resp
        if m: ans = m[1]

        print(f"\n\n> AI: '{ans}'\n")

    except InterruptedError:
        sys.exit()
    except Exception as e:
        print("Exception caught", e)
        traceback.print_exc()