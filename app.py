import os
import time
import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain import HuggingFacePipeline

import gradio as gr

def parse_args(print_args=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-large-en",
        help="Type of Huggingface embedding model"
    )
    parser.add_argument(
        "--data_path",
        default="data/pet.pdf",
        type=str,
        help="File where data is stored. Can be single pdf or jsonl file",
    )
    parser.add_argument(
        "--language_model",
        type=str,
        default="llm/zephyr-7b-beta.Q5_K_S.gguf",
        # choices = ["zephyr-7b-beta.Q5_K_S.gguf"],
        help="Type of Huggingface large language model"
    )
    parser.add_argument(
        "--database_path",
        type=str,
        default="db/vector.db",
        help="Location to store vector database",
    )
    parser.add_argument(
        "--num_documents",
        type=int,
        default=1,
        help="Number of relevant documents returned by semantic search",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Type of device. Can also be cuda"
    )
    parser.add_argument(
        "--ui",
        default=False, 
        action="store_true",
        help="Flag to enable UI via Gradio"
    )  
    parser.add_argument(
        "--ctransformers",
        default=False, 
        action="store_true",
        help="Flag to enable CTransformers"
    )

    parser.add_argument("--cache_dir", type=str)
    
    args = parser.parse_args()
    if print_args: print(vars(args))
    return args



def get_embedding_model(embedding_model: str = "BAAI/bge-large-en", device: str ="cpu", **kwargs):
    # Set pre-trained embedder 
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def get_language_model(language_model="", device="", cache_dir="", ctransformers=True, **kwargs):
    #TODO: does not work for GPU yet
    config = {
        "max_new_tokens": 1024,
        "repetition_penalty": 1.1,
        "temperature": 0.1,
        "top_k": 50,
        "top_p": 0.9,
        "stream": True,
        "threads": int(os.cpu_count() / 2)
    }

    
    
    if ctransformers:
        # Cpu compatible smaller transformers
        if "zephyr" in language_model or "mistral" in language_model:
            model_type = "mistral"
        else:
            raise NotImplementedError
        language_model = CTransformers(
            model=language_model,
            model_type=model_type,
            lib="avx2" if device == "cpu" else None
            **config
        )
    else:
        # Default huggingface transformers
        model = AutoModelForCausalLM.from_pretrained(
            language_model,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(language_model)

        generator = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            max_new_tokens=258,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        language_model = HuggingFacePipeline(pipeline=generator)
    return language_model


def initialize_qa(args):

    prompt_template = """<s>{context}

You are a helpdesk employee of SURF, answering questions for users on `servicedesk.surf.nl`. Above is text which might be helpful for your response. NO URLs except when they are in the above text. Below is a question which requires answering. Please be helpful and answer the question to the best of your ability. Be brief and accurate, remove any `http` from your output. 

Question: {question}

Answer:
    """

    print("Loading embedding model")
    embeddings = get_embedding_model(**vars(args))
    print("Loading large language model")
    language_model = get_language_model(**vars(args))
    

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    load_vector_store = Chroma(persist_directory=args.database_path, embedding_function=embeddings)

    from ingest import load_documents
    documents = load_documents(args.data_path)
    bm25_retriever = BM25Retriever.from_documents(documents)
    vec_retriever = load_vector_store.as_retriever(search_kwargs={"k": args.num_documents})

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vec_retriever],
        weights=[0.5,0.5]
    )

    return language_model, prompt, retriever

def bash_retrieval(language_model, prompt, retriever):
    query = input("Enter query: ")

    semantic_search = retriever.get_relevant_documents(query)
    print("Relevant document(s): \n:", semantic_search)

    print("Type of llm:", type(language_model))

    start_time = time.time()
    qa = RetrievalQA.from_chain_type(
        llm=language_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents = True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )

    response = qa(query)
    print(f"Retrieval took: {time.time() - start_time} seconds")
    print("Response: ", response["result"])
    return response



def launch_gradio(language_model, prompt, retriever):
    
    def get_response(input):
        start_time = time.time()
        query = input
        chain_type_kwargs = {"prompt": prompt}
        qa = RetrievalQA.from_chain_type(llm=language_model, chain_type="stuff", retriever=retriever, return_source_documents=False, chain_type_kwargs=chain_type_kwargs, verbose=True)
        response = qa(query)
        print(f"Retrieval took: {time.time() - start_time} seconds")
        return response["result"]

    input = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )

    sample_prompts = ["Here are some example queries"]
    iface = gr.Interface(fn=get_response, 
                inputs=input, 
                outputs="text",
                title="LLM QA",
                description="LLM QA",
                examples=sample_prompts,
                allow_flagging="never"
                )

    iface.launch()

def main(args):    
    start_time = time.time()
    print("Initializing model and retriever takes: ", time.time() - start_time)
    language_model, prompt, retriever = initialize_qa(args)

    if args.ui:
        launch_gradio(language_model, prompt, retriever)
    else:
        while True:
            bash_retrieval(language_model, prompt, retriever)


if __name__ == "__main__":
    main(parse_args())



            