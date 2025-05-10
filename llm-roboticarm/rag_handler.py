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
    """
    Handles Retrieval-Augmented Generation (RAG) by processing documents, creating embeddings,
    and setting up a retrieval mechanism using the OpenAI API and vector storage.
    """

    def __init__(self, files: list[tuple[str, str]], api_key: str):
        """
        Initialize the RAGHandler with multiple files.

        Parameters
        ----------
        files : list[tuple[str, str]]
            List of tuples containing (file_path, file_type).
        api_key : str
            The API key for accessing the OpenAI service.
        """
        self.files = files
        self.api_key = api_key
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embedding_model = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=self.api_key)
        self.vector_db = self._initialize_vector_db()
        self.rag_chain = self._initialize_rag_chain()

    def _check_file_exists(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

    def _extract_text_from_pdf(self, file_path):
        text = ''
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        return text

    def _extract_text_from_txt(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def _extract_text(self, file_path, file_type):
        if file_type == 'pdf':
            return self._extract_text_from_pdf(file_path)
        elif file_type == 'txt':
            return self._extract_text_from_txt(file_path)
        else:
            raise ValueError("Unsupported file type. Only 'pdf' and 'txt' files are supported.")

    def _load_and_split_documents(self):
        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        for file_path, file_type in self.files:
            self._check_file_exists(file_path)
            text = self._extract_text(file_path, file_type.lower())
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
        return [Document(page_content=chunk) for chunk in all_chunks]

    def _initialize_vector_db(self):
        documents = self._load_and_split_documents()
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            collection_name="multi-file-rag"
        )

    def _setup_retriever(self):
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are a robot agent named 'UR5e' in a human-robot collaborative assembly system. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome limitations of distance-based similarity search.
            Provide these alternative questions separated by newlines.
            Original question: {question}
            Your response must be clear and concise. Do NOT exceed 50 words."""
        )
        return MultiQueryRetriever.from_llm(
            retriever=self.vector_db.as_retriever(),
            llm=self.llm,
            prompt=QUERY_PROMPT
        )

    def _define_rag_prompt(self):
        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """
        return ChatPromptTemplate.from_template(template)

    def _initialize_rag_chain(self):
        retriever = self._setup_retriever()
        prompt = self._define_rag_prompt()
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def retrieve(self, question: str):
        return self.rag_chain.invoke({"question": question})