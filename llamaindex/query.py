import os, shutil
import logging
import sys
import chromadb
import argparse
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.chroma import ChromaVectorStore


def load(load_files=False):
    """
    LOADER

    Connects to the database and loads data if loads_files is true
    Return index
    """
    persist_dir = os.getenv("PERSIST_DIR")
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection("llama_index")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # uses open ai embedding algorithm, builds index
    # INDEX
    index = VectorStoreIndex.from_vector_store(vector_store,
                                               storage_context=storage_context, show_progress=True)

    if load_files:
        print("Loading new files from data/input")
        documents = SimpleDirectoryReader("data/input").load_data()
        for doc in documents:
            index.insert(doc)

    return index


def query(input_query, load_files):
    print("Querying index")
    index = load(load_files)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_query)
    print(response)


def get_parser():
    parser = argparse.ArgumentParser(description='Running LlamaIndex')
    parser.add_argument("function", type=str, help='function')
    parser.add_argument('-i', '--input', type=str, help='Input to the model')
    parser.add_argument('-l', '--load', action="store_true", help='Load new files')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.function == "clear_db":
        # delete all data in storage
        if os.path.isdir("storage"):
            shutil.rmtree("storage")
            return
    elif args.function == "query":
        if not args.input:
            print("No input query provided")
            return
        query(args.input, load_files=args.load)
    elif args.function == "test":
        print("Test")


if __name__ == "__main__":
    load_dotenv()
    log_path = os.getenv("LOGGER")
    logging.basicConfig(filename=log_path, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG)
    main()
