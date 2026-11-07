import os
os.environ["USER_AGENT"] = "Mozilla/5.0"


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

from website_urls import urls


# ----------------------------
# Load PDFs
# ----------------------------

def load_pdfs(data_folder="data"):

    all_docs = []

    for bank in os.listdir(data_folder):

        bank_path = os.path.join(data_folder, bank)

        if not os.path.isdir(bank_path):
            continue

        for file in os.listdir(bank_path):

            if file.lower().endswith(".pdf"):

                pdf_path = os.path.join(bank_path, file)

                loader = PyPDFLoader(pdf_path)

                docs = loader.load()

                for doc in docs:

                    doc.metadata["bank"] = bank
                    doc.metadata["source_file"] = file
                    doc.metadata["source_type"] = "pdf"

                all_docs.extend(docs)

    return all_docs


# ----------------------------
# Load Websites
# ----------------------------

def load_websites():

    all_docs = []

    for url in urls:

        try:

            loader = WebBaseLoader(url)

            docs = loader.load()

            for doc in docs:

                doc.metadata["source_type"] = "web"
                doc.metadata["url"] = url

                # bank name tagging

                if "sbi" in url.lower():
                    doc.metadata["bank"] = "SBI"

                elif "hdfc" in url.lower():
                    doc.metadata["bank"] = "HDFC"

                elif "bankofbaroda" in url.lower():
                    doc.metadata["bank"] = "BOB"

                else:
                    doc.metadata["bank"] = "Unknown"

            all_docs.extend(docs)

        except Exception as e:

            print(f"Error loading {url}: {e}")

    return all_docs


# ----------------------------
# Load Everything
# ----------------------------

def load_all_documents():

    pdf_docs = load_pdfs()

    web_docs = load_websites()

    all_docs = pdf_docs + web_docs

    print(f"Loaded {len(pdf_docs)} PDF pages")
    print(f"Loaded {len(web_docs)} Web documents")
    print(f"Total loaded: {len(all_docs)}")

    return all_docs