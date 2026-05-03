import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
import os

os.environ["HF_TOKEN"] = "Your Hugging Face Token Here"  

# ==========================================
# PAGE CONFIGURATION (CẤU HÌNH TRANG WEB)
# ==========================================
st.set_page_config(page_title="FinErva Stock Analyzer", page_icon="📈", layout="centered")

st.title("AI Stock Analysis Assistant (LLaVA-LoRA)")
st.markdown("Upload a price chart (Candlestick) to receive market trend insights.")

# ==========================================
# 1. MODEL LOADING FUNCTION (HÀM LOAD MODEL)
# ==========================================
@st.cache_resource(show_spinner="Loading AI...")
def load_model():
    model_id = "llava-hf/llava-1.5-7b-hf"
    adapter_path = "./llava-stock-analyzer/final"

    # Load Processor
    processor = AutoProcessor.from_pretrained(adapter_path, local_files_only=True)

    # Load Base Model 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0}
    )

    # Attach LoRA Adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval() 
    
    return processor, model

# Call the function to load the model
processor, model = load_model()
st.success("AI System is ready!")

# ==========================================
# 2. USER INTERFACE (GIAO DIỆN NGƯỜI DÙNG)
# ==========================================
# Upload component
uploaded_file = st.file_uploader("Upload (PNG, JPG)", type=["png", "jpg", "jpeg"])

# Text input component
user_prompt = st.text_area(
    "What do you want to ask about this chart?", 
    value="Based on this chart, what is the next price trend and why?",
    height=100
)

# Execution button
if st.button("Start Analysis", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a chart image first!")
    else:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input Chart", use_column_width=True)
        
        # Start inference
        with st.spinner("AI is analyzing the chart..."):
            
            # Format prompt (ĐÃ ĐƯỢC DỊCH SANG TIẾNG ANH)
            full_prompt = f"USER: <image>\nContext: \nQuestion: {user_prompt}\nASSISTANT:"
            
            inputs = processor(
                text=full_prompt, 
                images=image, 
                return_tensors="pt"
            ).to("cuda", torch.bfloat16)

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=250,
                    do_sample=False
                )

            # Decode the result
            response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract Assistant's answer
            final_answer = response.split("ASSISTANT:")[-1].strip()
            
            # Display the result
            st.divider()
            st.subheader("AI Insight:")
            st.info(final_answer)