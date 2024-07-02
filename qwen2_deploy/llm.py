from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

collection_name = "documents"
embedding_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
max_new_tokens = 300
task = "text-generation"
model = "Qwen/Qwen2-7B-Instruct"
cache_path = "/root/qdrant_cache"

hf = HuggingFacePipeline.from_model_id(
    model_id=model,
    task=task,
    pipeline_kwargs={
        "max_new_tokens": max_new_tokens,
    },
    model_kwargs={
        "temperature": 0.3,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
        "load_in_4bit": True,
        # "device_map": "auto",
    },
)

embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
vectorstore = Qdrant.from_existing_collection(
    embedding=embeddings,
    path=cache_path,
    collection_name=collection_name)

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """请根据以下背景知识回答问题. 如果背景知识中缺少相关的信息, 你可以根据经验进行回答, 但务必要基于你知道的事实而不是捏造的内容.

背景知识:

{context}

问题:

{question}

回答:

"""

prompt = PromptTemplate.from_template(template)
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | hf.bind()
)

import sys
import re
import readline
import traceback
sys.stdin.reconfigure(encoding='utf-8')
ans_pat = "^.*回答:[ \\n]*(.*)$"
print("\n\n")

while True:
    try:
        print("输入你的问题:\n")
        q = None
        while not q: q = sys.stdin.readline().strip()
        print(".. ")

        resp = chain.invoke(q)
        m = re.search(ans_pat, resp, re.DOTALL)
        ans = resp
        if m: ans = m[1]

        # print(f"\n{resp}\n")
        print(f"\n> {ans}\n")
    except InterruptedError:
        sys.exit()
    except Exception as e:
        print("Exception caught", e)
        traceback.print_exc()