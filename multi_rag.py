from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import requests
import tempfile
import uvicorn
app = FastAPI()
llm = ChatGroq(
    api_key="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",
    model_name="gemma2-9b-it",
    temperature=0
)

doc_store = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    global doc_store
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name


        pdf_loader = PyPDFLoader(tmp_file_path)
        pdf = pdf_loader.load()

        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(pdf)

    
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        doc_store = FAISS.from_documents(embedding=embedding_model, documents=docs)

        return "PDF uploaded and processed successfully."
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    global doc_store
    if doc_store is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded. Please upload a PDF first.")
    try:
        retriever = doc_store.as_retriever()

        
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentences maximum and keep the answer concise. "
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        
        doc_chain = create_stuff_documents_chain(prompt=prompt, llm=llm)
        main_chain = create_retrieval_chain(retriever, doc_chain)

       
        answer = main_chain.invoke({"input": query.question})

        
        final_answer = answer['answer']

        
        return {"answer": final_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)