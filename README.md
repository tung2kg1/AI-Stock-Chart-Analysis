# FinErva: AI Stock Chart Analyzer

FinErva is an intelligent stock analysis tool designed to interpret financial candlestick charts using advanced vision-language models. By combining **LLaVA (Large Language-and-Vision Assistant)** with specialized fine-tuning, the system provides automated technical analysis, pattern recognition, and market sentiment insights directly from chart images.

## Key Features

*   **Visual Pattern Recognition:** Automatically detects common candlestick patterns such as Head and Shoulders, Double Bottoms, and Bullish/Bearish Engulfing patterns.
*   **LLaVA-LoRA Integration:** Utilizes a **Low-Rank Adaptation (LoRA)** fine-tuned LLaVA model to understand the specific visual language of financial markets.
*   **Technical Indicator Context:** Interprets overlays like Moving Averages, RSI, and MACD when present on the uploaded chart.
*   **Automated Insights:** Generates human-like summaries of price action, support/resistance levels, and potential trend reversals.
*   **Interactive Dashboard:** A Streamlit-powered interface for seamless image uploads and real-time AI inference.

## Model & Architecture

### The FinErva Model
The core of this project is a **LLaVA-v1.5-7B** model specifically adapted for financial chart interpretation.

*   **Base Model:** LLaVA (Large Language-and-Vision Assistant), which connects a vision encoder (CLIP) with a language model (Vicuna/Llama).
*   **Fine-tuning (LoRA):** Instead of retraining the entire model, we use LoRA to update a small subset of weights. This allows the model to learn "financial literacy"—understanding what a green/red candle represents and how volume correlates with price—without losing its general reasoning capabilities.
*   **Precision:** Optimized for 4-bit or 8-bit quantization to run efficiently on consumer-grade GPUs (using compatible execution providers).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tung2kg1/AI-Stock-Chart-Analysis.git](https://github.com/tung2kg1/AI-Stock-Chart-Analysis.git)
    cd AI-Stock-Chart-Analysis
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    
```bash
    pip install -r requirements.txt
```

4.  **Model Weights:**
    Ensure you have the fine-tuned LoRA weights in the `llava-stock-analyzer/final/` directory or configured via environment variables.
    Download: https://drive.google.com/file/d/1FWkcXQezW_zFddQ6Qrg-wDQz7rTL0q2M/view?usp=sharing

## Usage

To start the analyzer:
```bash
streamlit run app.py
```

##  Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. 
The stock market involves significant risk, and AI models can produce "hallucinations" or incorrect interpretations. 
Always perform your own due diligence or consult a certified financial advisor before making investment decisions.
