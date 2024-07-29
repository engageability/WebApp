import  openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
#from langchain_community.embeddings import  HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from utils.constants import *




class PdfQA:
    def __init__(self,openai_api_key, huggingface_api_key, config:dict = {}):
        self.openai_api_key = openai_api_key
        self.huggingface_api_key = huggingface_api_key
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None


    # The following class methods are useful to create global GPU model instances
    # This way we don't need to reload models in an interactive app,
    # and the same model instance can be used across multiple user sessions

    def create_mpnet_base_v1():
        return HuggingFaceEmbeddings(model_name=EMB_MPNET_BASE_V1)
        

    #@classmethod
    def create_llama3_8B_instruct(self,temp = 0.01, max_new_tokens = 128):
        return HuggingFaceEndpoint(huggingfacehub_api_token=self.huggingface_api_key,
                     repo_id=LLM_LLAMA3_INSTRUCT, temperature=temp, max_new_tokens=max_new_tokens)


    def init_embeddings(self) -> None:
        if self.config["embedding"] == EMB_MPNET_BASE_V1:
            self.embedding = PdfQA.create_mpnet_base_v1()
        else:
            self.embedding = None 

    def init_models(self) -> None:
        """ Initialize LLM models based on config """
        if (self.config["llm"] == LLM_OPENAI_GPT35) or (self.config["llm"] == LLM_OPENAI_GPT4O) or (self.config["llm"] == LLM_OPENAI_GPT4O_MINI) or (self.config["llm"] == LLM_OPENAI_GPT4):
            openai.api_key = self.openai_api_key
            pass
        elif self.config["llm"] == LLM_LLAMA3_INSTRUCT:
            if self.llm is None:
                self.llm = PdfQA.create_llama3_8B_instruct(self)
        else:
            raise ValueError("Invalid config") 
               

    
    def vector_db_pdf(self) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """

        pdf_path = self.config.get("pdf_path",None)
        if pdf_path:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
   
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embedding)
        else:
            raise ValueError("NO PDF found")

    def retreival_qa_chain(self):
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        self.qa = None
        self.retriever = None

        if not self.vectordb:
            raise ValueError("Vector DB not initialized. Call vector_db_pdf() first.")
        
        self.retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        if (self.config["llm"] == LLM_OPENAI_GPT35) or (self.config["llm"] == LLM_OPENAI_GPT4O) or (self.config["llm"] == LLM_OPENAI_GPT4O_MINI) or (self.config["llm"] == LLM_OPENAI_GPT4):
            self.qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name=self.config["llm"], temperature=0),
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )

        else:
            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                
            )

    def answer_query(self, st, question: str) -> str:
        """
        Answer the question
        """
        if not self.qa:
            raise ValueError("QA chain not initialized. Call retreival_qa_chain() first.")
        
        try:
            answer_dict = self.qa({"query": question})
            st.write("Raw answer_dict:", answer_dict)
            
            if "result" in answer_dict:
                return answer_dict
            else:
                st.error("No 'result' key in the answer dictionary")
                return f"Error: Unexpected response format. Keys in response: {list(answer_dict.keys())}"
        except Exception as e:
            st.error(f"Error in answer_query: {str(e)}")
            return f"Error: {str(e)}"
        


