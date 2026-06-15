# 📈 MarketLens

### AI-Powered Stock Market Analytics, RAG & Natural Language SQL Assistant

MarketLens is a GenAI-powered analytics platform that combines Natural Language Processing, SQL generation, Retrieval-Augmented Generation (RAG), and interactive business intelligence dashboards.

Users can ask questions in plain English and receive:

* 🤖 AI-generated SQL queries
* 📊 Interactive Plotly dashboards
* 📈 KPI analytics
* 📚 RAG-powered document answers
* 🗄️ PostgreSQL insights
* 📄 PDF, Excel, and CSV exports

Built using LangGraph, FastAPI, PostgreSQL, FAISS, and a locally hosted Qwen LLM.

---

# 🚀 Features

## 🤖 Natural Language → SQL

Ask questions like:

* Which country has the highest average market capitalization?
* Show top 10 companies by market cap.
* Compare Technology and Healthcare sectors.
* Which stocks have the highest trading volume?

The AI automatically:

1. Understands the request
2. Generates SQL
3. Validates query safety
4. Executes against PostgreSQL
5. Returns analytical insights

---

## 📚 Retrieval-Augmented Generation (RAG)

MarketLens includes a semantic document assistant powered by:

* FAISS Vector Database
* Sentence Transformers
* Local Qwen LLM

Users can query:

* Financial regulations
* Market rules
* Investment strategies
* Internal company documents
* Research reports
* Policy documents

with source-aware responses.

---

## 📊 Interactive Dashboard

Auto-generated dashboards include:

* KPI Cards
* Bar Charts
* Line Charts
* Pie Charts
* Scatter Plots

Features:

* Cross-filtering
* Dynamic chart switching
* Interactive drill-down
* Responsive layout
* Dark/Light mode

---

## 📈 Financial Analytics

Supports:

* Market Capitalization Analysis
* PE Ratio Analysis
* Dividend Yield Analysis
* Sector Performance
* Country Comparison
* Exchange Analysis
* Trading Volume Insights

---

## 🧠 Conversational Memory

Supports follow-up questions:

Example:

User:

```text
Show top countries by market capitalization
```

Follow-up:

```text
Which one has the highest PE ratio?
```

The assistant automatically uses previous context.

---

# 🏗️ System Architecture

```text
User Question
      │
      ▼
 ┌─────────────┐
 │  FastAPI    │
 └─────────────┘
      │
      ▼
 ┌─────────────┐
 │ LangGraph   │
 │   Agent     │
 └─────────────┘
      │
 ┌────┴────┐
 ▼         ▼
SQL       RAG
Engine    Engine
 │         │
 ▼         ▼
Postgres  FAISS
 │         │
 └────┬────┘
      ▼
 Local Qwen LLM
      ▼
Final Response
      ▼
Plotly Dashboard
```

---

# 🛠️ Tech Stack

| Layer           | Technology            |
| --------------- | --------------------- |
| Backend         | FastAPI               |
| Agent Framework | LangGraph             |
| LLM             | Qwen 4B               |
| Database        | PostgreSQL            |
| Embeddings      | Sentence Transformers |
| Vector Store    | FAISS                 |
| Visualization   | Plotly                |
| Frontend        | HTML, CSS, JavaScript |
| Authentication  | Bearer Token          |
| Deployment      | Windows / Linux       |

---

# 📂 Project Structure

```text
MarketLens/
│
├── gateway/
│   ├── main.py
│   ├── auth.py
│   ├── llm_manager.py
│   └── context_memory.py
│
├── tools/
│   ├── sql.py
│   ├── rag.py
│   └── chart.py
│
├── docs/
│   └── RAG source documents
│
├── csv_file/
│
├── IMAGES_PDF/
│
├── index.html
├── run.py
├── requirements.txt
├── .env.example
└── README.md
```

---

# 📊 Supported Stock Market Dataset

MarketLens currently supports datasets using the following schema:

| Column             | Description           |
| ------------------ | --------------------- |
| stock_id           | Stock Identifier      |
| ticker             | Stock Symbol          |
| company_name       | Company Name          |
| country            | Country               |
| sector             | Business Sector       |
| exchange           | Stock Exchange        |
| currency           | Currency              |
| trade_date         | Trading Date          |
| open_price         | Opening Price         |
| high_price         | Highest Price         |
| low_price          | Lowest Price          |
| close_price        | Closing Price         |
| adjusted_close     | Adjusted Close        |
| volume             | Trading Volume        |
| market_cap_billion | Market Capitalization |
| pe_ratio           | Price Earnings Ratio  |
| dividend_yield     | Dividend Yield        |

---

# 💬 Example Questions

## SQL Analytics

* Show all stocks.
* Show top 10 companies by market cap.
* Which country has the most companies?
* Average PE ratio by sector.
* Highest volume stocks.
* Compare exchanges by market capitalization.
* Show dividend yield by country.

---

## Financial Analysis

* What is PE ratio?
* Explain market capitalization.
* What is dividend yield?
* What is CAGR?
* Difference between bull and bear market.

---

## RAG Questions

* Explain stock market regulations in India.
* What are SEC reporting requirements?
* Explain NSE trading rules.
* How does T+1 settlement work?
* What is insider trading?

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/AvanindraVijay/MarketLens.git

cd MarketLens
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Create:

```bash
.env
```

Example:

```env
DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=marketdb
DB_PORT=5432

DOCS_DIR=docs
VECTORSTORE_DIR=faiss_index
TOP_K_CHUNKS=5

CONTEXT_MEMORY_ENABLED=true
```

---

## Run Application

```bash
python run.py
```

Open:

```text
http://127.0.0.1:8000
```

---

# 🔒 Authentication

Development users:

| User        | Token            |
| ----------- | ---------------- |
| arjun_singh | dev-token-arjun  |
| priya_nair  | dev-token-priya  |
| vikram_rao  | dev-token-vikram |

Replace with JWT/OAuth before production deployment.

---

# 📡 API Endpoints

| Endpoint                  | Description        |
| ------------------------- | ------------------ |
| GET /                     | Dashboard          |
| GET /health               | Health Check       |
| GET /model_status         | LLM Status         |
| POST /chat                | Main Chat Endpoint |
| POST /threads/new         | Create Thread      |
| GET /threads              | List Threads       |
| GET /threads/{id}/history | Thread History     |
| POST /charts/export       | Export Charts      |

---

# 📸 Screenshots

Add screenshots inside:

```text
IMAGES_PDF/
```

Examples:

```md
![Dashboard](IMAGES_PDF/dashboard.png)

![Charts](IMAGES_PDF/charts.png)

![Chat](IMAGES_PDF/chat.png)
```

---

# 🚀 Key Highlights

✅ LangGraph Multi-Agent Workflow

✅ Natural Language to SQL

✅ PostgreSQL Analytics

✅ FAISS Vector Search

✅ Retrieval-Augmented Generation (RAG)

✅ Local Qwen LLM

✅ Plotly Dashboards

✅ Dynamic KPI Generation

✅ Interactive Charts

✅ PDF/Excel/CSV Export

✅ Conversational Memory

✅ Stock Market Intelligence

---

# 📈 Future Enhancements

* Multi-database support
* Live stock market APIs
* Portfolio tracking
* Real-time alerts
* Agentic workflows
* Multi-user authentication
* Cloud deployment
* Streamlit integration
* Fine-tuned financial LLM

---

# 🤝 Contributing

Pull requests, issues, and feature suggestions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a pull request

---

# 📜 License

MIT License

---

# 👨‍💻 Author

**Avanindra Vijay**

AI Engineer | GenAI Developer | Data Scientist

GitHub: https://github.com/AvanindraVijay

LinkedIn: https://linkedin.com/in/vijayavanindra

---

⭐ If you found this project useful, consider starring the repository.
