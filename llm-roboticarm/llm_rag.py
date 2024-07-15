import os
import openai

import pdfplumber

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document  # Import Document

# Load the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("No API key for OpenAI found in the environment variables.")
openai.api_key = OPENAI_API_KEY

class LLMRAG:
    def __init__(self):
        # Initialize the embedding model with your OpenAI API key
        self.embedding_model = OpenAIEmbeddings(openai_api_key=openai.api_key)

        # Initialize the language model with your OpenAI API key
        self.llm = ChatOpenAI(model="gpt-4", openai_api_key=openai.api_key)
        self.file_path = None
        
        self.index = None
        self.text_chunks = []
        self.chunk_size = 1000  # Define chunk size
        self.chunk_overlap = 200  # Define chunk overlap
        self.file_path = None

        self.memory = ConversationBufferMemory()

    def process_specification_file(self, file_path):
        self.file_path = file_path if file_path else "SOP2.pdf"
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        
        #loader = UnstructuredPDFLoader(file_path=self.file_path)
        #data = loader.load()
        try:
            with pdfplumber.open(self.file_path) as pdf:
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
        except Exception as e:
            print(f"An error occurred while reading the PDF: {e}")
            return
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_text(text)

        # Convert text chunks to Document objects
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Generate embeddings using a pre-trained model
        embedding_model = OpenAIEmbeddings()

        # Convert document chunks into a vector database
        vector_db = Chroma.from_documents(
            documents=documents, 
            embedding=embedding_model,
            collection_name="local-rag"
        )

        # Define the retrieval prompt
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}"""
        )

        print('jonghan000')
        print(QUERY_PROMPT)
        print('jonghan111')

        # Set up the retriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_db.as_retriever(), 
            llm=ChatOpenAI(model="gpt-4"),
            prompt=QUERY_PROMPT
        )

        # Define the RAG prompt
        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        # Create the processing chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | ChatOpenAI(model="gpt-4o")
            | StrOutputParser()
        )

        # Ask the first question and store the response in conversation memory
        response = chain.invoke({"question": "What is this about?"})
        self.memory.save_context({"input": "What is this about?"}, {"output": response})
        print(response)
        print("==========================================================================================")

        # Retrieve and append history for the second question
        history = self.memory.load_memory_variables({})['history']
        second_question = f"{history}\nHuman: What was my first question?\nAI:"
        print('jonghan1')
        print(second_question)
        print('jonghan2')
        print(history)
        print('jonghan3')
        response2 = chain.invoke({"question": second_question})
        self.memory.save_context({"input": "What was my first question?"}, {"output": response2})
        print(response2)


if __name__ == "__main__":
    llm_rag = LLMRAG()
    data = llm_rag.process_specification_file("C:\\Users\\jongh\\projects\\llm-roboticarm\\llm-roboticarm\\SOP2.pdf")

