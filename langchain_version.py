import os
from pathlib import Path

from pypdf import PdfReader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# Same corpus PDF as eval/rag_core.py (see eval/run_eval.py CORPUS_PDF).
PDF_PATH = Path(__file__).resolve().parent / "eval" / "corpus" / "adani_hindenburg_sc_2024.pdf"


def load_pdf_text(path: Path) -> str:
    # Same extraction as eval/rag_core.py load_pdf_text (and app.py).
    reader = PdfReader(str(path))
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n\n"
    return full_text.strip()


if __name__ == "__main__":
    text = load_pdf_text(PDF_PATH)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    print(f"Created {len(chunks)} chunks")
    print("\nFirst chunk:")
    print(chunks[0])

    # Same embedding model as eval/rag_core.py's default RagConfig.embed_model.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    question = "Who chaired the Expert Committee constituted by the Supreme Court?"
    retrieved = retriever.invoke(question)

    print(f"\nQuery: {question}")
    print(f"Retrieved {len(retrieved)} chunks:\n")
    for i, doc in enumerate(retrieved, 1):
        print(f"[{i}] {doc.page_content}\n")

    # Same key source as app.py: os.environ.get("GROQ_API_KEY") — app.py also
    # checks st.secrets first, which only applies inside a Streamlit app.
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        raise SystemExit(
            "GROQ_API_KEY not set. Set it in your environment "
            '(e.g. $env:GROQ_API_KEY="gsk_...") before running this script.'
        )

    llm = ChatGroq(model="openai/gpt-oss-20b", api_key=groq_api_key)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below.\n"
        "Context: {context}\n\n"
        "Question: {question}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    print(f"\nGenerated answer:\n{answer}")
