# EDRIC User and Execution Manual

## 1. Quickstart Guide

### Prerequisites
- Python 3.9+ (Python 3.10+ recommended)
- Google Gemini API Key or OpenAI API Key

### Installation
1. Clone the repository:
   git clone https://github.com/SardarAhmed05/Alphatron-Technologies-Research-Internship.git
   cd Batch 2/Week 4

2. Create and activate a virtual environment:
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   source venv/bin/activate  # Linux/macOS

3. Install dependencies:
   pip install -r requirements.txt

4. Configure .env file:
   Copy .env.example to .env and set your GOOGLE_API_KEY.

---

## 2. Launching the Streamlit Web Application
Execute the following command in your terminal:
   streamlit run app.py

Features available in the Web UI:
- Live URL Scraping mode (enter any web URL)
- Live Topic Search mode (enter any research question)
- Direct Text/HTML paste mode
- Interactive AgGrid data tables with search and column sorting
- 1-Click download buttons for CSV, Excel (.xlsx), and JSON
- Real-time trust score meters and verification badges

---

## 3. Launching the Terminal CLI
Run the interactive CLI interface:
   python cli.py

Supported CLI Commands:
- /scrape <url> : Scrapes and verifies a webpage
- /search <topic> : Executes live search and intelligence extraction
- /export <csv|json> : Exports extracted records to ./exports/
- /status : Displays session statistics
- exit : Quits the CLI session

---

## 4. Running the Master 5-Step Pipeline
To run all 5 OOP steps sequentially:
   python Main.py sample_data/sample_ecommerce.html

Or run individual standalone steps:
   python Step_1_WebFetcher.py
   python Step_2_ExtractorAgent.py
   python Step_3_ValidationGraph.py
   python Step_4_DataExporter.py
   python Step_5_ScraperEvaluator.py

---

## 5. Running Automated Tests
Run the full pytest suite:
   pytest -v