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
    def __init__(self, file_path, file_type, api_key):
        """
        Initialize the RAGHandler.

        Parameters
        ----------
        file_path : str
            The path to the file to be processed.
        file_type : str
            The type of the file ('pdf' or 'txt').
        api_key : str
            The API key for accessing the OpenAI service.
        """
        self.file_path = file_path
        self.file_type = file_type.lower()
        self.api_key = api_key
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embedding_model = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=self.api_key)
        self.rag_chain = self.initialize_rag_chain()

    def _check_file_exists(self):
        """
        Checks if the specified file exists.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """        
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

    def _extract_text_from_pdf(self):
        """
        Extracts text content from a PDF file.

        Returns
        -------
        str
            The extracted text from the PDF, with each page separated by a newline.
        """        
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

    def _extract_text_from_txt(self):
        """
        Reads and returns text from a plain text file.

        Returns
        -------
        str
            The entire content of the text file.
        """        
        text = ''
        try:
            with open(self.file_path, 'r') as file:
                text = file.read()
        except Exception as e:
            print(f"An error occurred while reading the text file: {e}")
        return text

    def _extract_text(self):
        """
        Extracts text based on file type.

        Returns
        -------
        str
            The extracted text.

        Raises
        ------
        ValueError
            If the file type is unsupported.
        """        
        if self.file_type == 'pdf':
            return self._extract_text_from_pdf()
        elif self.file_type == 'txt':
            return self._extract_text_from_txt()
        else:
            raise ValueError("Unsupported file type. Only 'pdf' and 'txt' files are supported.")

    def _split_text_into_chunks(self, text):
        """
        Splits text into smaller chunks for processing.

        Parameters
        ----------
        text : str
            The text to split.

        Returns
        -------
        list
            A list of text chunks.
        """        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return text_splitter.split_text(text)

    def _convert_chunks_to_documents(self, chunks: list) -> list:
        """
        Converts text chunks into Document objects.

        Parameters
        ----------
        chunks : list
            List of text chunks.

        Returns
        -------
        list
            A list of Document objects created from the chunks.
        """        
        return [Document(page_content=chunk) for chunk in chunks]

    def _create_vector_db(self, documents):
        """
        Creates a vector database from a list of documents.

        Parameters
        ----------
        documents : list
            List of Document objects.

        Returns
        -------
        Chroma
            A Chroma vector database with embeddings from the documents.
        """        
        return Chroma.from_documents(
            documents=documents, 
            embedding=self.embedding_model,
            collection_name="local-rag"
        )

    def _setup_retriever(self, vector_db):
        """
        Sets up a retriever with multi-query capabilities to enhance document retrieval.

        Parameters
        ----------
        vector_db : Chroma
            The vector database for document embeddings.

        Returns
        -------
        MultiQueryRetriever
            The retriever configured for multi-query processing.
        """        
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
        """
        Defines the prompt template for the RAG process.

        Returns
        -------
        ChatPromptTemplate
            A chat prompt template for RAG processing.
        """      

        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """
        return ChatPromptTemplate.from_template(template)

    def _create_processing_chain(self, retriever, prompt):
        """
        Creates a processing chain for RAG by combining retriever, prompt, and LLM response.

        Parameters
        ----------
        retriever : MultiQueryRetriever
            The retriever to fetch relevant documents.
        prompt : ChatPromptTemplate
            The prompt for formatting the response.

        Returns
        -------
        dict
            The processing chain for the RAG workflow.
        """        
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | ChatOpenAI(model="gpt-4o")
            | StrOutputParser()
        )

    def initialize_rag_chain(self):
        """
        Initializes the RAG processing chain by extracting text, creating documents,
        setting up the vector database, and defining the retriever and prompt.

        Returns
        -------
        dict
            The complete RAG processing chain.
        """        
        self._check_file_exists()  # Check if the file exists
        text = self._extract_text()  # Extract text based on file type
        chunks = self._split_text_into_chunks(text)  # Split text into manageable chunks
        documents = self._convert_chunks_to_documents(chunks)  # Convert chunks to Document objects
        vector_db = self._create_vector_db(documents)  # Create vector database with embeddings
        retriever = self._setup_retriever(vector_db)  # Setup retriever with multi-query capability
        prompt = self._define_rag_prompt()  # Define prompt template for LLM responses
        return self._create_processing_chain(retriever, prompt)  # Create and return the processing chain

    def retrieve(self, question: str):
        """
        Retrieves a response based on the user's question by invoking the RAG processing chain.

        Parameters
        ----------
        question : str
            The user's question to retrieve relevant information.

        Returns
        -------
        str
            The generated response based on the provided context.
        """        
        response = self.rag_chain.invoke({"question": question})
        return response