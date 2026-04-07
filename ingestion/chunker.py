from langchain_text_splitters import RecursiveCharacterTextSplitter

def text_split(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)