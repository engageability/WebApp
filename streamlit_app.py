import streamlit as st
import pandas as pd
import base64, os, re
from utils.constants import *
from utils.pdf_qa import PdfQA
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
except KeyError as e:
    st.error(f"Could not find {e} in secrets. Have you set it up correctly?")
    st.stop()

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['HUGGINGFACE_API_KEY'] = HUGGINGFACE_API_KEY

# Streamlit app code
st.set_page_config(
    page_title='Report Analysis Tool',
    page_icon='🌿',
    layout='wide',
    initial_sidebar_state='auto',
)


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("static/images/nature-sustainable-background.jpeg")


page_bg_img = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

body {{
    font-family: 'Poppins', sans-serif;
    color: #2c3e50;
}}

[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{img}");
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-size: cover;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-color: #e0f2f1;
    background-image: url("data:image/svg+xml,%3Csvg width='100' height='20' viewBox='0 0 100 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M21.184 20c.357-.13.72-.264 1.088-.402l1.768-.661C33.64 15.347 39.647 14 50 14c10.271 0 15.362 1.222 24.629 4.928.955.383 1.869.74 2.75 1.072h6.225c-2.51-.73-5.139-1.691-8.233-2.928C65.888 13.278 60.562 12 50 12c-10.626 0-16.855 1.397-26.66 5.063l-1.767.662c-2.475.923-4.66 1.674-6.724 2.275h6.335zm0-20C13.258 2.892 8.077 4 0 4V2c5.744 0 9.951-.574 14.85-2h6.334zM77.38 0C85.239 2.966 90.502 4 100 4V2c-6.842 0-11.386-.542-16.396-2h-6.225zM0 14c8.44 0 13.718-1.21 22.272-4.402l1.768-.661C33.64 5.347 39.647 4 50 4c10.271 0 15.362 1.222 24.629 4.928C84.112 12.722 89.438 14 100 14v-2c-10.271 0-15.362-1.222-24.629-4.928C65.888 3.278 60.562 2 50 2 39.374 2 33.145 3.397 23.34 7.063l-1.767.662C13.223 10.84 8.163 12 0 12v2z' fill='%2380cbc4' fill-opacity='0.2' fill-rule='evenodd'/%3E%3C/svg%3E");
}}

.stApp {{
    background-color: rgba(255, 255, 255, 0.7);
}}

h1, h2, h3 {{
    color: #1a5f7a;
}}

.stButton>button {{
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}}

.stButton>button:hover {{
    background-color: #45a049;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}}

.stTextInput>div>div>input {{
    background-color: #f1f8e9;
    border-radius: 4px;
}}

.custom-info-box {{
    background-color: #e7f3fe;
    border-left: 6px solid #2196F3;
    margin-bottom: 15px;
    padding: 16px;
    border-radius: 4px;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 24px;
}}

.stTabs [data-baseweb="tab"] {{
    height: 60px;
    white-space: pre-wrap;
    background-color: transparent !important;
    border-radius: 4px 4px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}}

.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
    font-size: 24px;
    font-weight: 600;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: #4CAF50 !important;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: #4CAF50 !important;
}}
[data-testid="stHeader"] {{
    background-color: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}

</style>
"""


st.markdown(page_bg_img, unsafe_allow_html=True)



if "pdf_qa_model" not in st.session_state:
    st.session_state["pdf_qa_model"]:PdfQA = PdfQA(openai_api_key=OPENAI_API_KEY, huggingface_api_key=HUGGINGFACE_API_KEY)  # Initialisation

#@st.cache_resource
def load_llm(llm):
    if (llm == LLM_OPENAI_GPT35) or (llm == LLM_OPENAI_GPT4O) or (llm == LLM_OPENAI_GPT4O_MINI):
        pass
    elif llm == LLM_LLAMA3_INSTRUCT:
        return PdfQA.create_llama3_8B_instruct()
    else:
        raise ValueError("Invalid LLM setting")

#@st.cache_resource
def load_emb(emb):
    if emb == EMB_MPNET_BASE_V1:
        return PdfQA.create_mpnet_base_v1()
    else:
        raise ValueError("Invalid embedding setting")


def categorize_response(response):
    response_lower = response.lower()
    if re.search(r'\byes\b', response_lower):
        return "Yes"
    elif re.search(r'\bno\b', response_lower):
        return "No"
    else:
        return "Not Given"


st.title("Report Analysis Tool")

with st.sidebar:
    st.header("Configuration")


    emb = EMB_MPNET_BASE_V1
    llm = st.radio("**Select LLM Model**", [LLM_OPENAI_GPT35,LLM_OPENAI_GPT4O_MINI,LLM_OPENAI_GPT4O,LLM_OPENAI_GPT4, LLM_LLAMA3_INSTRUCT], index=1)
    pdf_file = st.file_uploader("**Upload PDF**", type="pdf")

    if st.button("Submit") and pdf_file is not None:
        with st.spinner(text="Processing PDF and Generating Embeddings.."):
            try:

                pdf_path = os.path.join(os.path.dirname(__file__), pdf_file.name)
                
                with open(pdf_path, "wb") as f:
                    f.write(pdf_file.getbuffer())
                
                st.session_state["pdf_qa_model"].config = {
                    "pdf_path": pdf_path,
                    "embedding": emb,
                    "llm": llm
                } 
                st.session_state["pdf_qa_model"].init_embeddings()
                st.session_state["pdf_qa_model"].init_models()
                st.session_state["pdf_qa_model"].vector_db_pdf()
                st.session_state["pdf_qa_model"].retreival_qa_chain()
                st.sidebar.success("PDF processed successfully")
            except Exception as e:
                st.error(f"An error occurred: {e}")



if "pdf_file_name" in st.session_state:
    st.write(f"Currently loaded PDF: {st.session_state['pdf_file_name']}")

# Create two tabs
tab1, tab2 = st.tabs(["Batch Q&A", "Interactive Q&A"])

# Tab 1: Batch Q&A
with tab1:
    st.header("Batch Q&A")
    questions_file = st.file_uploader("Upload a CSV file with questions", type="csv")
    
    if questions_file is not None:
        questions_df = pd.read_csv(questions_file,encoding='unicode_escape')
        st.write("Preview of uploaded questions:")
        st.write(questions_df.head())
        
        if st.button("Process Batch Questions"):
            if "pdf_qa_model" in st.session_state and hasattr(st.session_state["pdf_qa_model"], "answer_query"):
                answers = []
                relevant_pages  = []
                for question in questions_df['Questions']:
                
                    result = st.session_state["pdf_qa_model"].answer_query(st, question + 'Give me an answer: yes or no answer and reasoning from the context')
                    answers.append(result["result"])
                    relevant_pages.append([doc.metadata.get("page", None) for doc in result["source_documents"]])

                questions_df['Response'] = answers
                questions_df['Relevant_pages'] = list(relevant_pages)
                questions_df["Answer_model"] = questions_df["Response"].apply(categorize_response)
                
                st.write("Results:")
                st.write(questions_df)
                
                # Option to download results
                csv = questions_df.to_csv(index=False)


                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="qa_results.csv",
                    mime="text/csv",
                )
            else:
                st.error("Please upload and process a PDF file first.")

# Tab 2: Interactive Q&A
with tab2:
    st.header("Interactive Q&A")
    question = st.text_input('Ask a question', 'What is this document?')

    if st.button("Answer"):
        try:
            st.session_state["pdf_qa_model"].retreival_qa_chain()
            answer = st.session_state["pdf_qa_model"].answer_query(st, question)
        except Exception as e:
            st.error(f"Error answering the question: {str(e)}")

eco_tips = [
    "Turn off lights when you leave a room to save energy.",
    "Use a reusable water bottle instead of disposable plastic ones.",
    "Try carpooling or using public transport to reduce carbon emissions.",
    "Plant a tree or start a small garden to support local biodiversity.",
    "Reduce meat consumption to lower your carbon footprint.",
]

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Eco Tip of the Day")
st.sidebar.info(eco_tips[pd.Timestamp.now().day % len(eco_tips)])

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for a sustainable future.")
