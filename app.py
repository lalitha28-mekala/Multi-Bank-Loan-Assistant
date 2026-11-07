import os
import streamlit as st

from dotenv import load_dotenv

from loaders import load_all_documents

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_classic.chains import RetrievalQA


from langchain_core.prompts import PromptTemplate



# ==========================
# Environment
# ==========================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY missing in .env file")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY



# ==========================
# Streamlit Config
# ==========================

st.set_page_config(
    page_title="Bank Loan AI Assistant",
    page_icon="🏦",
    layout="wide"
)


st.title("🏦 AI Powered Bank Loan Assistant")

st.write(
    """
    Ask questions about SBI, HDFC and Bank of Baroda
    Home Loans and Loan Against Property.
    """
)



# ==========================
# Embedding Model
# ==========================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )


embeddings = load_embeddings()



# ==========================
# Chroma DB
# ==========================

persist_directory = "chroma_db"



@st.cache_resource
def get_vector_db():

    if not os.path.exists(persist_directory) or len(os.listdir(persist_directory)) == 0:

        st.info("Creating Vector Database...")

        documents = load_all_documents()


        if not documents:
            st.error("No documents found")
            st.stop()


        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )


        chunks = splitter.split_documents(documents)


        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )


        return vectordb


    else:

        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )



vectordb = get_vector_db()



# ==========================
# Retriever
# ==========================

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k":5
    }
)



# ==========================
# Prompt
# ==========================


template = PROMPT_TEMPLATE = """
You are an AI Powered Bank Loan Assistant.

Your job is to answer customer questions using ONLY the retrieved information from SBI, HDFC, and Bank of Baroda documents.

Instructions:

1. Understand the user's question carefully.
2. Give a direct and concise answer first.
3. Compare banks only when the user asks for comparison.
4. Show only the relevant information related to the question.
5. Do NOT include irrelevant fields.
6. Do NOT repeat the same information.
7. Do NOT mention "uploaded documents" or "context".
8. If information is missing, say:
   "The available documents do not provide this information."
9. Use a simple banking advisor tone.
10. If multiple banks match the query, provide a comparison table.
11. Do not make assumptions outside the provided context.
QUERY UNDERSTANDING AND ABBREVIATIONS:

1. Understand common banking abbreviations used by customers.

2. Interpret the following abbreviations:

   - HL = Home Loan
   - Home Loans = Home Loan
   - LAP = Loan Against Property
   - BOB = Bank of Baroda
   - SBI = State Bank of India
   - HDFC = HDFC Bank

3. Customers may use short forms, spelling variations,
   or informal language.

4. Understand the intended meaning of the query
   before generating the answer.

5. Treat abbreviations and their corresponding full forms
   as equivalent when interpreting the question.

Retrieved Context:
{context}

User Question:
{question}

Answer Format:

### Answer
<direct answer>

### Comparison (Only if needed)
| Feature | SBI | HDFC | Bank of Baroda |
|----------|------|------|------|
| ... | ... | ... | ... |


"""


PROMPT = PromptTemplate(
    template=template,
    input_variables=[
        "context",
        "question"
    ]
)



# ==========================
# Gemini LLM
# ==========================


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)



# ==========================
# RAG Chain
# ==========================


qa_chain = RetrievalQA.from_chain_type(

    llm=llm,

    retriever=retriever,

    chain_type="stuff",

    return_source_documents=True,

    chain_type_kwargs={
        "prompt":PROMPT
    }

)



# ==========================
# User Input
# ==========================


query = st.text_input(
    "Ask your question:",
    placeholder="Compare SBI and HDFC home loans"
)



if st.button("Submit"):


    if query:


        with st.spinner("Searching documents..."):


            response = qa_chain.invoke(
                {
                    "query":query
                }
            )


        answer = response["result"]

        documents = response["source_documents"]


        st.subheader("Answer")

        st.write(answer)



        
        st.divider()

        st.subheader("📚 Sources")

        already_seen = set()

        source_count = 0
        max_sources = 2

        for doc in documents:

            metadata = doc.metadata

            bank = metadata.get("bank", "Unknown")
            source_type = metadata.get("source_type", "Unknown")
            source_file = metadata.get("source_file", "")
            url = metadata.get("url", "")

            # Use URL for websites and filename for PDFs
            source_key = url if url else source_file

            if not source_key:
                continue

            # Skip duplicate sources
            if source_key in already_seen:
                continue

            already_seen.add(source_key)

            st.write(f"**Bank:** {bank}")

            st.write(f"**Source Type:** {source_type}")

            if source_file:
                st.write(f"**File:** {source_file}")

            if url:
                st.markdown(f"**Source:** [{url}]({url})")

            st.divider()

            source_count += 1

            # Stop after displaying two unique sources
            if source_count >= max_sources:
                break