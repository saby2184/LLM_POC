import streamlit as st
from langchain.prompts import PromptTemplate
#from langchain_community.llms import CTransformers
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import torch

##call doc
##def get_resp(doc,query):
    ## LLAMA model2



#create the UI

#st.set_page_config(page_title="Generate Blogs"
#                )
#st.head("Generate Blogs")
#input_text=st.text_input("Attach the document ")
#submit = st.button("Ask")

# Final call
#if submit:
#    st.write(get_resp(doc,query))

### Prompt template

template = """Question: {question}

Answer: Explain the answer briefly."""

prompt = PromptTemplate.from_template(template)


db_dir="./chroma_db"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.



#Vector store
embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
    )
db = Chroma(persist_directory=db_dir, embedding_function=embeddings)




# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="local_llm_model/llama-2-7b-chat.Q8_0.gguf",
    n_gpu_layers=n_gpu_layers,
    temperature=0.75,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)
#print(llm.invoke("Simulate a rap battle between Stephen Colbert and John Oliver") )


retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    #chain_type_kwargs={"prompt": prompt}, 
    retriever=retriever, 
    verbose=True
)


query = """
Question: what is multihead attention
"""
#print(llm.invoke(prompt))

result = qa({"query": query})
print(result)
