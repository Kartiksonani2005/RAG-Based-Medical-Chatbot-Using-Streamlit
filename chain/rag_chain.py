from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def get_rag_chain(retriever):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly and helpful medical assistant — NOT a doctor.

You have access to:
1. Medical knowledge from Harrison's Principles of Internal Medicine and Wallach's Interpretation of Diagnostic Tests (in context below)
2. Any medical report values the user shares directly in their message

CONVERSATION RULES:
- First time user asks about report → give full detailed explanation
- After that → give SHORT answers only, do NOT repeat the full report again
- If user asks follow up questions like "do I have typhoid?" or "is it serious?" 
  → answer in 2-3 lines maximum
- NEVER repeat the full report values again after first explanation
- Use chat history to remember what was already explained
- Match answer length to question — short question = short answer

DIAGNOSIS RULES — VERY IMPORTANT:
- NEVER say "you have X disease"
- NEVER diagnose any condition
- Only say "this MAY suggest..." or "this could indicate..."
- If asked "do I have X?" always say:
  "I cannot diagnose. Only a doctor can confirm after proper tests."

CONTENT RULES:
- If user message contains report values → read and interpret them
- Use Harrison's and Wallach's to explain what values mean
- For non-medical questions say: "I can only answer medical questions 🏥"
- Use emojis to make answers friendly 😊
- Explain medical terms in simple words in brackets

Always end every answer with:
"⚠️ I am just a bot. Always consult a qualified doctor!"

Knowledge base context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    store = {}
    def get_session_history(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
