# AI-Powered Multi-Bank Loan Assistant using RAG

## 1. Project Overview

The **AI-Powered Multi-Bank Loan Assistant** is a Generative AI application developed using Retrieval-Augmented Generation (RAG).

The application helps users understand and compare loan-related information from three banks:

- State Bank of India (SBI)
- HDFC Bank
- Bank of Baroda (BOB)

It focuses on two loan categories:

- Home Loan (HL)
- Loan Against Property (LAP)

Users can ask questions in natural language and receive answers based on information retrieved from bank PDFs and official websites.

The application uses Google Gemini to generate responses using the retrieved information, helping reduce unsupported or irrelevant answers.

## 2. Problem Statement

Customers applying for home loans or loans against property often need to visit different bank websites and search through multiple documents to understand:

- Interest rates
- Loan eligibility criteria
- Required documents
- Processing fees and charges
- Repayment conditions
- Loan-related terms and conditions

Finding and comparing this information manually can be time-consuming.

This project aims to simplify the process by providing a single conversational interface where users can ask loan-related questions and retrieve relevant information from multiple bank sources.

## 3. Project Objectives

- Develop a conversational loan information assistant.
- Retrieve information from multiple bank documents and websites.
- Support Home Loan and Loan Against Property queries.
- Generate context-based responses using a Large Language Model.
- Reduce the need for manually searching multiple documents.
- Provide source information along with generated answers.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | User interface |
| LangChain | RAG pipeline development |
| PyPDFLoader | Loading PDF documents |
| WebBaseLoader | Loading website content |
| RecursiveCharacterTextSplitter | Splitting documents into chunks |
| Hugging Face Embeddings | Converting text into vector representations |
| BAAI/bge-small-en-v1.5 | Embedding model |
| ChromaDB | Vector database |
| Google Gemini | Answer generation |
| Semantic Search | Retrieving relevant information |

## 5. Project Architecture and Workflow

The application follows a basic Retrieval-Augmented Generation architecture.

### Step 1: Data Collection

Collected loan-related information from:

- Bank PDF documents.
- Official bank websites.

The collected information covers SBI, HDFC Bank and Bank of Baroda.

### Step 2: Document Loading

Used LangChain document loaders:

- PyPDFLoader – to load PDF documents.
- WebBaseLoader – to extract content from websites.

### Step 3: Text Chunking

Used RecursiveCharacterTextSplitter to divide large documents into smaller chunks.

Configuration:

- Chunk Size: 1000
- Chunk Overlap: 200

Chunking helps divide documents into manageable pieces while maintaining some context between consecutive chunks.

### Step 4: Text Embeddings

Used the Hugging Face embedding model:

`BAAI/bge-small-en-v1.5`

The embedding model converts text chunks into numerical vector representations.

These vectors help identify semantically similar content during retrieval.

### Step 5: Vector Database

Used ChromaDB to store:

- Text chunks
- Vector embeddings
- Document metadata

Metadata includes information such as bank name, source type, source file and website URL.

The vector database is persisted locally for reuse.

### Step 6: Information Retrieval

Implemented semantic similarity search using the ChromaDB retriever.

When a user asks a question:

1. The question is converted into an embedding.
2. The retriever searches for semantically similar document chunks.
3. Relevant chunks are returned as context.

### Step 7: Answer Generation

Integrated Google Gemini using LangChain's RetrievalQA chain.

The retrieved document chunks are passed to Gemini as context.

The prompt instructs the model to:

- Answer using the retrieved information.
- Avoid unsupported assumptions.
- Provide a clear and concise response.
- Indicate when the required information is unavailable in the retrieved context.

### Step 8: User Interface

Developed a Streamlit interface where users can:

- Enter loan-related questions.
- Receive generated answers.
- View retrieved source information.

## 6. High-Level Architecture

```text
       PDF Documents + Bank Websites
                    |
                    v
             Document Loading
                    |
                    v
               Text Chunking
                    |
                    v
             Hugging Face
                Embeddings
                    |
                    v
                ChromaDB
             Vector Database
                    |
                    |
User Question ------+
       |
       v
Question Embedding
       |
       v
Semantic Similarity Search
       |
       v
Relevant Document Chunks
       |
       v
   Gemini LLM
       |
       v
 Generated Answer
       |
       v
    Streamlit UI
       |
       v
Answer + Source Information
```

## 7. Current Implementation

The following features are implemented in the current version:

- PDF document loading.
- Website content extraction.
- Recursive text splitting.
- Hugging Face text embeddings.
- ChromaDB vector storage.
- Semantic similarity-based retrieval.
- RetrievalQA chain integration.
- Google Gemini answer generation.
- Prompt-based response grounding.
- Streamlit user interface.
- Manual evaluation of generated responses.

## 8. Evaluation

Currently, I perform manual evaluation by testing different loan-related questions and checking:

- Whether the answer is relevant to the question.
- Whether the answer matches the retrieved document context.
- Whether the retrieved sources contain the required information.
- Whether the model generates unsupported information.
- Whether the response is clear and understandable.

This helps identify retrieval and answer-generation issues.

## 9. Future Enhancements

The current application implements a basic RAG pipeline using semantic similarity search. I plan to improve the retrieval quality, context handling and evaluation process in future versions.

### 9.1 Hybrid Search

Currently, the application uses semantic search.

In the future, I plan to implement hybrid search by combining:

- Semantic search using vector embeddings.
- Keyword-based search using BM25.

This can help retrieve information based on both the meaning of a question and exact keywords, such as bank names, loan types, percentages and specific financial terms.

### 9.2 Context-Aware Retrieval

I plan to explore context-aware retrieval techniques to improve the quality of retrieved information.

These techniques can use additional document context to make chunks more meaningful and improve retrieval accuracy.

### 9.3 Improved Chunking Strategies

Explore advanced chunking approaches, such as:

- Contextual chunking.
- Parent-child chunking.
- Semantic chunking.

The objective is to preserve important relationships between document sections and reduce the loss of context during retrieval.

### 9.4 Query Enhancement

Implement query transformation techniques such as:

- Query rewriting.
- Query expansion.
- Multi-query retrieval.
- Abbreviation and terminology expansion.

These techniques can help the system understand different ways users express the same loan-related question.

### 9.5 Improved Retrieval and Reranking

Explore:

- Dynamic Top-K retrieval.
- Metadata-based filtering.
- Hybrid retrieval.
- Reranking models.
- Context compression.

These improvements can help identify the most relevant information before passing it to the LLM.

### 9.6 Standard RAG Evaluation

Currently, evaluation is performed manually.

In future versions, I plan to introduce standard RAG evaluation techniques and metrics using frameworks such as **RAGAS**.

Planned evaluation metrics include:

| Metric | Purpose |
|---|---|
| Faithfulness | Checks whether the generated answer is supported by the retrieved context. |
| Answer Relevancy | Measures how relevant the answer is to the user's question. |
| Context Precision | Checks whether the retrieved context contains relevant information. |
| Context Recall | Measures whether the retrieval process captures the important information needed to answer the question. |

I also plan to evaluate retrieval performance using metrics such as:

- Hit Rate / Recall@K.
- Mean Reciprocal Rank (MRR).
- Precision@K.

These metrics can help evaluate retrieval quality separately from answer-generation quality.

## 10. Learning Outcomes

Through this project, I gained practical experience in:

- Understanding Retrieval-Augmented Generation.
- Working with PDF and website document loaders.
- Text chunking and embeddings.
- Vector database integration using ChromaDB.
- Semantic similarity search.
- Integrating Large Language Models using LangChain.
- Developing an interactive application using Streamlit.
- Understanding the importance of grounding LLM responses in retrieved information.

## 11. Conclusion

This project demonstrates a basic multi-bank loan information assistant using Retrieval-Augmented Generation.

It combines document loading, text chunking, embeddings, vector storage, semantic retrieval and LLM-based answer generation into a single application.

The current version establishes a working RAG baseline, with future improvements planned for hybrid retrieval, context-aware techniques and systematic evaluation.
