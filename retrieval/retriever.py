def get_retriever(docsearch):
    return docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15}
    )