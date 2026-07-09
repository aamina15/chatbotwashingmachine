import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="Samsung AI Manual Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(to right,#eef7ff,#ffffff);
}

.title{
font-size:42px;
font-weight:bold;
text-align:center;
color:#0b5ed7;
}

.subtitle{
text-align:center;
color:gray;
font-size:18px;
margin-bottom:20px;
}

[data-testid="stSidebar"]{
background:#0b5ed7;
}

[data-testid="stSidebar"] *{
color:white;
}

.stChatMessage{
padding:15px;
border-radius:15px;
margin-bottom:10px;
box-shadow:0px 2px 8px rgba(0,0,0,.15);
}

.footer{
text-align:center;
margin-top:40px;
color:gray;
font-size:15px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
"""
<div class='title'>
🤖 Samsung Washing Machine AI Assistant
</div>

<div class='subtitle'>
Ask questions about your Samsung Washing Machine Manual.
</div>
""",
unsafe_allow_html=True
)

# ---------------- Sidebar ----------------

st.sidebar.title("📘 AI Manual Chatbot")

st.sidebar.markdown("### Features")

st.sidebar.success("HTML Manual")
st.sidebar.success("OpenAI GPT")
st.sidebar.success("LangChain")
st.sidebar.success("ChromaDB")
st.sidebar.success("Semantic Search")

st.sidebar.markdown("---")

st.sidebar.info("""
Example Questions

• Child Lock

• Eco Bubble

• Error 5C

• Clean Filter

• Delay End

• Drum Clean
""")

# ---------------- Load Model ----------------

@st.cache_resource
def load_vectorstore():

    loader = UnstructuredHTMLLoader(
        "How to use the various modes of the washing machine _ Samsung LEVANT.html"
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    return vectorstore


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are an expert Samsung Washing Machine assistant.

Answer ONLY from the provided manual.

If the answer cannot be found, reply:

"I couldn't find that information in the manual."

Manual:

{context}

Question:

{question}

Answer:
""")

chain = prompt | llm | StrOutputParser()

# ---------------- Chat History ----------------

if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- User Input ----------------

question = st.chat_input("Ask your question...")

if question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching Manual..."):

            docs = retriever.invoke(question)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            answer = chain.invoke(
                {
                    "context":context,
                    "question":question
                }
            )

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )

st.markdown(
"""
<div class="footer">
Made with ❤️ using Streamlit | LangChain | ChromaDB | OpenAI
</div>
""",
unsafe_allow_html=True
)
