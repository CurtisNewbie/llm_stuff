from langchain_core.prompts import PromptTemplate
import sys
import readline
from langchain_huggingface import HuggingFacePipeline

hf = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 150,
    },
    model_kwargs={
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
    },
)

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf.bind()

while True:
    try:
        print("Enter your question:")
        q = sys.stdin.readline().strip()
        if not q: continue
        resp = chain.invoke({"question": q})
        print(f"\n\n> model: '{resp}'\n")

    except InterruptedError:
        sys.exit()
    except Exception as e:
        print(f"Exception caught {e}")