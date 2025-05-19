# Chat Log Parser and Summarizer

A Python tool to analyze and summarize chat logs from `.txt` files using NLP techniques. Extracts key topics, message statistics, and generates summaries.



## Features
- Parse single or multiple chat log files
- Extract speaker-specific messages (User/AI)
- Identify main topics using TF-IDF and lemmatization
- Generate summary statistics (message counts, keywords)

## Prerequisites
- Python 3.12.4
- pip 24.0+

## Setup

### 1. Create and activate virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet punkt
```

## Usage

### 1. For Single Chat File
```bash
python ai_chat_summarize_for_single_txt_file.py
```

**Output Example**:
```text
Total messages: 4
User messages: 2
AI messages: 2

 Summary
 - The conversation had 15 exchanges
 - The user asked mainly about python and use
 - Most common keywords: python, use, hi, tell, sure
```
[for_single_file](assets\for_single_file.png) 

### 2. For Multiple Chat Files
```bash
python ai_chat_summarize_to_parse_all_txt_and_analysis.py
```

**Output Example**:
```text
Total messages: 8
User messages: 4
AI messages: 4

Summary
The conversation had 26 exchanges
The user asked mainly about python and ai
Most common keywords: python, ai, data, hi, learn
```
[mltiple_txt_parse](assets\mltiple_txt_parse.png) 

### 3. Jupyter Notebook Option
```bash
jupyter notebook AI_Chat_Log_Summarizer_multiple_txt_parse.ipynb
```
[for_ipynb](assets\for_ipynb.png) 

## Adding Screenshots
1. Create an `assets/` folder:
   ```bash
   mkdir assets
   ```
2. Save screenshot (e.g., `sample_output.png`) in this folder


## Project Structure
```
.
├── chat_log/                  # Folder for input chat logs (.txt files)
├── venv/                      # Virtual environment (ignored)
├── assets/                    # For screenshots and images
├── .gitignore
├── requirements.txt
├── README.md
├── ai_chat_summarize_for_single_txt_file.py
├── ai_chat_summarize_to_parse_all_txt_and_analysis.py
└── AI_Chat_Log_Summarizer_multiple_txt_parse.ipynb
```

## Technical Details
- Uses NLTK for tokenization and lemmatization
- TF-IDF vectorization for keyword extraction
- Regular expression pattern matching for message parsing:
  ```python
  PATTERN = r'(User|AI):\s*(.*?)(?=\n*User:|\n*AI:|\$)'
  ```

## Troubleshooting
- If you get NLTK errors, re-run:
  ```bash
  python
  >>> import nltk
  >>> nltk.download('stopwords') 
  and so on (necessary libraries)
  ```
- For virtual environment issues:
  ```bash
  deactivate
  ```
