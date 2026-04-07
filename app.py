import os
import streamlit as st
from dotenv import load_dotenv

from embedding.embedder import get_embedding
from vectorstore.pinecone_client import get_pinecone_vectorstore
from retrieval.retriever import get_retriever
from chain.rag_chain import get_rag_chain
from ingestion.pdf_loader import process_uploaded_pdf
from ingestion.ocr_loader import process_uploaded_image

load_dotenv()

st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
[data-testid="stToolbar"] {display: none;}
.stDeployButton {display: none;}
footer {display: none;}
#MainMenu {display: none;}
.block-container {
    padding-top: 2rem;
    padding-bottom: 0rem;
    max-width: 800px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "user1"
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "uploaded_report_text" not in st.session_state:
    st.session_state.uploaded_report_text = ""
if "ask_doctor" not in st.session_state:
    st.session_state.ask_doctor = False
if "chain" not in st.session_state:
    with st.spinner("Loading medical knowledge base..."):
        try:
            embedding = get_embedding()
            docsearch = get_pinecone_vectorstore(embedding)
            retriever = get_retriever(docsearch)
            st.session_state.chain = get_rag_chain(retriever)
            st.session_state.docsearch = docsearch
            st.session_state.embedding = embedding
        except Exception as e:
            st.error(f"❌ Failed to connect to knowledge base: {e}")
            st.stop()

# Header
st.markdown("<h2 style='text-align:center;'>🏥 Medical RAG Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Ask medical questions or upload your reports</p>", unsafe_allow_html=True)
st.divider()

# Chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Doctor suggestion
if st.session_state.ask_doctor:
    st.info("👨‍⚕️ Would you like to consult a doctor? Please visit your nearest hospital or clinic for professional medical advice.")

# File uploader
uploaded_files = st.file_uploader(
    "Attach files",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# Process files
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if new_files:
        with st.spinner(f"Processing {len(new_files)} file(s)..."):
            all_text = ""
            all_chunks = []

            for uploaded_file in new_files:
                try:
                    if uploaded_file.name.lower().endswith(".pdf"):
                        chunks = process_uploaded_pdf(uploaded_file)
                    else:
                        chunks = process_uploaded_image(uploaded_file)

                    if chunks:
                        file_text = "\n".join([c.page_content for c in chunks])
                        all_text += f"\n--- {uploaded_file.name} ---\n{file_text}"
                        all_chunks.extend(chunks)
                        st.session_state.processed_files.append(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} processed!")
                    else:
                        st.error(f"❌ No text extracted from {uploaded_file.name}")

                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

            if all_chunks:
                try:
                    st.session_state.docsearch.add_documents(all_chunks)
                    retriever = get_retriever(st.session_state.docsearch)
                    st.session_state.chain = get_rag_chain(retriever)
                except Exception as e:
                    st.warning(f"⚠️ Could not add to knowledge base: {e}")

            if all_text:
                st.session_state.uploaded_report_text += all_text

if st.session_state.processed_files:
    st.caption(f"📎 Ready: {', '.join(st.session_state.processed_files)}")

# Chat input
if prompt := st.chat_input("Ask a medical question..."):

    st.session_state.ask_doctor = False

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.uploaded_report_text:
                    full_input = f"""Please read and analyze this medical report:

{st.session_state.uploaded_report_text}

User question: {prompt}

Interpret all values, identify abnormal ones and explain what they mean using your medical knowledge from Harrison's and Wallach's."""
                else:
                    full_input = prompt

                response = st.session_state.chain.invoke(
                    {"input": full_input},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
                answer = response["answer"]

            except Exception as e:
                st.warning("🔄 Reconnecting to knowledge base...")
                try:
                    embedding = get_embedding()
                    docsearch = get_pinecone_vectorstore(embedding)
                    retriever = get_retriever(docsearch)
                    st.session_state.chain = get_rag_chain(retriever)
                    st.session_state.docsearch = docsearch

                    response = st.session_state.chain.invoke(
                        {"input": full_input},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    answer = response["answer"]
                except Exception as e2:
                    answer = f"Sorry I'm having trouble connecting. Please refresh the page. Error: {str(e2)}"

            st.markdown(answer)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.ask_doctor = True