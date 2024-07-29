# LLmReportAnalyzer
Preview

## Overview
This Streamlit application provides an interactive tool to analyze and extract insights from PDF reports using OpenAI or Meta's large language models (LLMs). It is designed for automated report analysis and QA tasks, offering a user-friendly interface to facilitate in-depth exploration of report contents.

## Features

### PDF Report Analysis
Perform detailed analysis of PDF reports. The application utilizes LLMs to extract key insights, trends, and summaries from the content.

### Interactive Interface
Engage with the data through an interactive Streamlit interface. The tool supports various visualization options to better understand the analyzed data.

### Customizable Analysis
Tailor the analysis according to your needs by adjusting settings and parameters. The application is flexible and scalable for different report analysis requirements.

## How to Use

1. **Select an LLM**
 Select an LLM in the sidebar configuration.

2. **Upload a PDF Report**
Use the file uploader in the sidebar to select and upload the PDF report you want to analyze.
  
3. **Submit Configuration and PDF**
Press the submit button once the model is selected and the PDF uploaded.

4. **Run QA Batch**
In the 'Batch Q&A' tab you can upload a CSV file with a batch of questions you want to ask and press process batch.
IMPORTANT: The column name in the CSV file needs to be 'Questions' for it to work properly.

5. **Download the result**
When the QA Batch is finished press download to download a CSV file with all the questions and answers.
     
6. **Interactive QA**
In the 'Interactive Q&A' tab you can ask single questions instead of whole batches.

## Data Source
The application processes PDF reports uploaded by the user. It uses OpenAI's API to perform the analysis or Huggingface's API paired with the [LLama3 8B Instruct model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).


## Usage
Deploy on Streamlit Cloud:
- Create an Account: Sign up at Streamlit Cloud.
- Connect to GitHub: Link your Streamlit account to your GitHub repository.
- Enter your OpenAI and Hugginface API keys into the Secrets under 'Advanced settings'
- Deploy the App: In Streamlit Cloud, select the repository and branch, then click "Deploy".


