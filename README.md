# Chat Log Parser and Summarizer

A Python tool to analyze and summarize chat logs from `.txt` files using NLP techniques. Extracts key topics, message statistics, and generates summaries.

![Sample Output](assets/sample_output.png) *(Example: Replace with your screenshot)*

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
Total messages: 42
User messages: 21
AI messages: 21

Summary
- The conversation had 18 exchanges
- The user asked mainly about python and installation
- Most common keywords: python, install, error, version, package
```

### 2. For Multiple Chat Files
```bash
python ai_chat_summarize_to_parse_all_txt_and_analysis.py
```

**Output Example**:
```text
Total messages: 127
User messages: 64
AI messages: 63

Summary
The conversation had 45 exchanges
The user asked mainly about api and debugging
Most common keywords: api, debug, error, response, timeout
```

### 3. Jupyter Notebook Option
```bash
jupyter notebook AI_Chat_Log_Summarizer_multiple_txt_parse.ipynb
```

## Adding Screenshots
1. Create an `assets/` folder:
   ```bash
   mkdir assets
   ```
2. Save your screenshot (e.g., `sample_output.png`) in this folder
3. Reference it in markdown:
   ```markdown
   ![Sample Output](assets/sample_output.png)
   ```

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
  python -m nltk.downloader stopwords wordnet punkt
  ```
- For virtual environment issues:
  ```bash
  deactivate && rm -rf venv/ && python -m venv venv
  ```

## License
MIT