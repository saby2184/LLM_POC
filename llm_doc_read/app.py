import streamlit as st
from langchain.prompts import PromptTemplate
#from langchain_community.llms import CTransformers
from langchain_community.llms import LlamaCpp


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




n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.








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

prompt = """
Question: Explain what is an transformers in large language models
"""
print(llm.invoke(prompt))
