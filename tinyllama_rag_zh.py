from langchain_core.prompts import PromptTemplate
import re
import traceback
import sys
import readline
from langchain_huggingface import HuggingFacePipeline

max_new_tokens=300
task="text-generation"
model="TinyLlama/TinyLlama_v1.1_chinese"
files = []

hf = HuggingFacePipeline.from_model_id(
    model_id=model,
    task=task,
    pipeline_kwargs={
        "max_new_tokens": max_new_tokens,
    },
    model_kwargs={
        # "max_length": 300,
        "trust_remote_code":True,
        "temperature": 0.7,
        "top_k": 10,
        "top_p": 0.95,
        "do_sample": True
    }
)

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

# load the document and split it into chunks
documents = []
for f in files: documents.extend(TextLoader(f).load())

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")

# load it into Chroma
vec = Chroma.from_documents(docs, embed)
# retri = vec.as_retriever() # default: k is 4, num of doc
# retri = vec.as_retriever(search_kwargs={"k": 2})
# retri = vec.as_retriever(search_type="mmr",search_kwargs={'k': 2, 'fetch_k': 5, 'lambda_mult': 0.5})
retri = vec.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """请根据以下背景知识回答问题. 如果背景知识中缺少相关的信息, 你可以根据经验进行回答, 但务必要基于你知道的事实而不是捏造的内容.

背景知识:

{context}

我的问题:

{question}

你的回答:

"""

prompt = PromptTemplate.from_template(template)

chain = (
    {"context": retri | format_docs, "question": RunnablePassthrough()}
    | prompt
    | hf.bind()
)

print("\n\n")
sys.stdin.reconfigure(encoding='utf-8')
ans_pat = "^.*你的回答:[ \\n]*(.*)$"

while True:
    try:
        print("输入你的问题:\n")
        q = None
        while not q: q = sys.stdin.readline().strip()
        print()

        resp = chain.invoke(q)
        m = re.search(ans_pat, resp, re.DOTALL)
        ans = resp
        if m: ans = m[1]

        print(f"\n{resp}\n")
        # print(f"\n{ans}\n")

    except InterruptedError:
        sys.exit()
    except Exception as e:
        print("Exception caught", e)
        traceback.print_exc()
