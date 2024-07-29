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

1. **Upload a PDF Report**
   - Use the file uploader in the sidebar to select and upload the PDF report you want to analyze.

2. **Configure Analysis Settings**
   - Adjust the analysis settings as needed. Options include selecting specific sections, adjusting summarization levels, and more.

3. **View Analysis Results**
   - Explore the results through various visualizations provided by the application. These include extracted text, key points, summaries, and trend analysis.

## Data Source
The application processes PDF reports uploaded by the user. It uses OpenAI's API to perform the analysis or Huggingface's API paired with the [LLama3 8B Instruct model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

## Installation

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/dataminer33/llmReportAnalyzer.git
   cd llmReportAnalyzer
   \`\`\`

2. Create a virtual environment and activate it:
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use \`venv\Scripts\activate\`
   \`\`\`

3. Install the required packages:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. Set up your OpenAI API key:
   - Add your OpenAI API key in a file named \`api_keys.py\`:
     \`\`\`python
     OPENAI_API_KEY = 'your_openai_api_key'
     \`\`\`

## Usage

1. Run the Streamlit application:
   \`\`\`bash
   streamlit run streamlit_app.py
   \`\`\`

2. Open your web browser and go to \`http://localhost:8501\` to interact with the application.

## Acknowledgments

- Thanks to the Streamlit community for creating an amazing tool for data visualization.
- Special thanks to OpenAI for providing the powerful language models used in this application.
