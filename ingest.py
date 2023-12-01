import argparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import JSONLoader


def parse_args(print_args=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="data/pet.pdf",
        type=str,
        help="File where data is stored. Can be single pdf or jsonl file",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-large-en",
        help="Type of Huggingface embedding model")
    
    parser.add_argument(
        "--database_path",
        type=str,
        default="db/vector.db",
        help="Location to store vector database",
    )
    parser.add_argument(
        "--bm25_database_path",
        type=str,
        default="db/bm25.db",
        help="Location to store full text search database",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Type of device. Can also be cuda"
    )
    
    args = parser.parse_args()
    if print_args: print(vars(args))
    return args

def get_embedding_model(embedding_model: str = "BAAI/bge-large-en", device: str ="cpu"):
    # Set pre-trained embedder 
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def load_documents(data_path: str = "pet.pdf"):
    # Load documents
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        jq_schema = ".text"
        loader = JSONLoader( 
            file_path=data_path,
            jq_schema=jq_schema,
            text_content=True,
            json_lines=data_path.endswith(".jsonl")) # If newline json file then set True
    elif data_path.endswith(".pdf"):
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(data_path)
    else:
        raise NotImplementedError
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    data = text_splitter.split_documents(data)
    print(f"I have {len(data)} lines")
    return data

def main(args):
    print("Loading embedding model")
    embeddings = get_embedding_model(embedding_model = args.embedding_model, device=args.device)
    print("Loading documents")
    documents = load_documents(args.data_path)

    print("Saving to Vector database")
    vector_db = Chroma.from_documents(documents, embeddings, persist_directory=args.database_path, collection_metadata={"hnsw:space": "cosine"})
    # Necessary only when calling in notebook where client object will be destroyed and database will be persisted anyway
    #vector_db.persist()
    print("Vector database created")

if __name__ == "__main__":
    main(parse_args())
