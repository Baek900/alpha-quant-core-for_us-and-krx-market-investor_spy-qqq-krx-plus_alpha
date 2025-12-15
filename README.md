# alpha-quant-core-for_us-and-krx-market-investor_spy-qqq-krx-plus_alpha
AI-powered Investment Advisor with Deep Learning Stock Prediction System using Hybrid LSTM-Transformer Architecture for NASDAQ(QQQ), S&amp;P500(SPY), and KOSPI. Built with PyTorch &amp; Streamlit.

# 🤖 Global AI Asset Advisor

> **Deep Learning-based Investment Signal Service for NASDAQ, S\&P 500, and KOSPI.**
> A subscription-based AI service that analyzes three major global markets to provide optimal daily investment positions.

   

## 📖 Project Overview

This project utilizes a **Hybrid Deep Learning Model** combining **LSTM (Long Short-Term Memory)** and **Transformer Attention** mechanisms to predict short-term stock market trends.

Going beyond simple price fluctuation predictions, the model detects market uncertainty and suggests three clear actionable signals: **'Buy', 'Sell', and 'Hold'**. The system is specifically optimized for an **Accumulation (Scaling-in) Strategy** using leveraged ETFs (2x, 3x).

## 🚀 Key Features

  * **🌍 Triple Market Coverage:** Full support for US Tech (QQQ), US Large Cap (SPY), and the Korean Market (KOSPI 200).
  * **🧠 Advanced AI Algorithm:** Multi-dimensional analysis of capital flow between sectors and technical indicators (RSI, Bollinger Bands, Volume Ratio, etc.).
  * **🔒 Subscription Membership:** Secure login system accessible only to approved subscribers.
  * **📱 Responsive Dashboard:** A user-friendly web interface built with Streamlit, accessible from both PC and mobile devices.
  * **💰 Actionable Trading Guide:** Provides specific instructions (e.g., "Buy $1,000 of QLD") rather than vague market views.

## 🛠️ Tech Stack

  * **Language:** Python 3.10+
  * **Deep Learning:** PyTorch (LSTM + Transformer Architecture)
  * **Web Framework:** Streamlit
  * **Data Source:** `yfinance` (US Markets), `pykrx` (Korean Market)
  * **Deployment:** Streamlit Community Cloud

## 📂 Project Structure

```bash
├── models/                     # Pre-trained AI Model files (.pth)
│   ├── us_sector_ai_model_qqq.pth
│   ├── us_spy_target_best_model.pth
│   └── kospi_model.pth
├── app.py                      # Main Web Application Code
├── model_def.py                # PyTorch Model Class Definition
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation
```

## 💻 How to Run Locally

Follow these steps to run the project on your local machine:

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/[Your-Username]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Secrets (For Login)**

      * Create a `.streamlit` folder in the root directory.
      * Create a `secrets.toml` file inside that folder.
      * Add your admin credentials:

    <!-- end list -->

    ```toml
    [users]
    admin = "password123"
    ```

4.  **Run the App**

    ```bash
    streamlit run app.py
    ```

## ☁️ Deployment

This service is deployed via **Streamlit Community Cloud**.
Any changes pushed to the GitHub repository are automatically deployed to the live website.

## ⚠️ Disclaimer

The AI analysis provided by this service is for informational purposes only and should not be considered absolute investment advice. All investment decisions and risks are the sole responsibility of the user. Investment involves risk.
