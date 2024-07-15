import os
import pdfplumber
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document

class RAGHandler:
    def __init__(self, file_path, api_key):
        self.file_path = file_path
        self.api_key = api_key
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embedding_model = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=self.api_key)
        self.rag_chain = self.initialize_rag_chain()

    def _check_file_exists(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

    def _extract_text_from_pdf(self):
        text = ''
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
        except Exception as e:
            print(f"An error occurred while reading the PDF: {e}")
        return text

    def _split_text_into_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return text_splitter.split_text(text)

    def _convert_chunks_to_documents(self, chunks):
        return [Document(page_content=chunk) for chunk in chunks]

    def _create_vector_db(self, documents):
        return Chroma.from_documents(
            documents=documents, 
            embedding=self.embedding_model,
            collection_name="local-rag"
        )

    def _setup_retriever(self, vector_db):
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}"""
        )
        return MultiQueryRetriever.from_llm(
            retriever=vector_db.as_retriever(), 
            llm=self.llm,
            prompt=QUERY_PROMPT
        )

    def _define_rag_prompt(self):
        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """
        return ChatPromptTemplate.from_template(template)

    def _create_processing_chain(self, retriever, prompt):
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | ChatOpenAI(model="gpt-4o")
            | StrOutputParser()
        )

    def initialize_rag_chain(self):
        self._check_file_exists()
        text = self._extract_text_from_pdf()
        chunks = self._split_text_into_chunks(text)
        documents = self._convert_chunks_to_documents(chunks)
        vector_db = self._create_vector_db(documents)
        retriever = self._setup_retriever(vector_db)
        prompt = self._define_rag_prompt()
        return self._create_processing_chain(retriever, prompt)

    def retrieve(self, question: str):
        response = self.rag_chain.invoke({"question": question})
        return response