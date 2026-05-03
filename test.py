import os
import json
import re
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# HuggingFace & Deep Learning
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from datasets import Dataset

# Metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Tải gói tokenizer của nltk cho việc tính BLEU Score
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ==========================================
# 1. TẢI CẤU HÌNH VÀ MÔ HÌNH (QLoRA)
# ==========================================
print("🔄 Đang thiết lập cấu hình và tải mô hình...")

# Sử dụng ID chuẩn từ HuggingFace để tự động tải weights nếu thiếu
base_model_id = "llava-hf/llava-1.5-7b-hf" 
adapter_path = "./llava-stock-analyzer/final" # Thư mục chứa trọng số LoRA bạn đã train

# 1.1 Load Processor
processor = AutoProcessor.from_pretrained(adapter_path)

# 1.2 Cấu hình lượng tử hóa 4-bit (Tránh tràn VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 # Dùng float16 an toàn trên Windows
)

# 1.3 Load Base Model 
print("⏳ Đang tải Base Model (Có thể mất thời gian nếu hệ thống cần tải file safetensors từ internet)...")
base_model = LlavaForConditionalGeneration.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# 1.4 Gắn Adapter LoRA vào Base Model
print("⏳ Đang gắn trọng số LoRA...")
model = PeftModel.from_pretrained(base_model, adapter_path)
print("✅ Tải hệ thống mô hình thành công!")

# ==========================================
# 2. TẢI VÀ TIỀN XỬ LÝ TẬP DỮ LIỆU ĐÁNH GIÁ
# ==========================================
print("\n🔄 Đang tải tập dữ liệu đánh giá (Validation Set)...")
current_dir = os.getcwd() 
dataset_dir = os.path.join(current_dir, "FinErva")

finerva_data_path = os.path.join(dataset_dir, "data-finerva-price", "data", "problems.json")
splits_path = os.path.join(dataset_dir, "data-finerva-price", "data", "pid_splits.json")
raw_image_folder = os.path.join(dataset_dir, "images-price")


with open(finerva_data_path, 'r', encoding='utf-8') as f:
    raw_problems = json.load(f)
with open(splits_path, 'r', encoding='utf-8') as f:
    splits_data = json.load(f)

eval_ids = splits_data.get("val", []) 

def extract_to_list(id_list):
    dataset_list = []
    for pid in id_list:
        if pid not in raw_problems:
            continue
            
        item = raw_problems[pid]
        image_filename = item.get("image", "image.png")
        original_img_path = os.path.join(raw_image_folder, pid, image_filename)
        
        if not os.path.exists(original_img_path):
            continue 
            
        choices = item.get("choices", [])
        answer_idx = item.get("answer", 0)
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        correct_answer = choices[answer_idx] if answer_idx < len(choices) else ""
        
        human_text = f"Ngữ cảnh: {item.get('lecture', '')}\n\nCâu hỏi: {item.get('question', '')}\nCác lựa chọn:\n{choices_text}\nHãy chọn đáp án đúng và giải thích."
        gpt_text = f"Đáp án đúng là: {correct_answer}.\nGiải thích: {item.get('solution', '')}"
        
        dataset_list.append({
            "id": pid,
            "image": original_img_path, 
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt", "value": gpt_text}
            ]
        })
    return dataset_list

eval_dataset = Dataset.from_list(extract_to_list(eval_ids))
print(f"✅ Tải dữ liệu thành công! Tìm thấy {len(eval_dataset)} mẫu hợp lệ.")

# ==========================================
# 3. CÁC HÀM ĐÁNH GIÁ (EVALUATION METRICS)
# ==========================================

def parse_llava_response(text):
    """Sử dụng Regex để bóc tách Đáp án (A/B/C/D) và Lời giải thích từ kết quả trả về"""
    # Tìm chữ cái A, B, C, D nằm sau cụm "Đáp án đúng là:"
    choice_match = re.search(r"Đáp án đúng là:\s*([A-D])", text, re.IGNORECASE)
    choice = choice_match.group(1).upper() if choice_match else "UNKNOWN"
    
    # Lấy toàn bộ phần văn bản sau chữ "Giải thích:"
    explanation_match = re.search(r"Giải thích:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else text.strip()
    
    return choice, explanation

def evaluate_model(model, processor, eval_dataset, num_samples=50):
    """Vòng lặp chạy Inference và tính toán các ma trận nhầm lẫn / độ đo NLP"""
    model.eval()
    
    true_choices, pred_choices = [], []
    true_explanations, pred_explanations = [], []
    
    print(f"\n🚀 Đang tiến hành chạy kiểm thử trên {num_samples} mẫu...")
    
    for i in tqdm(range(min(num_samples, len(eval_dataset)))):
        instance = eval_dataset[i]
        
        # 1. Trích xuất câu hỏi và đáp án gốc (Ground Truth)
        conv = instance['conversations']
        human_text = conv[0]['value'] 
        true_gpt_text = conv[1]['value']
        
        true_choice, true_exp = parse_llava_response(true_gpt_text)
        
        # 2. Xây dựng Prompt cho mô hình (Chỉ đưa câu hỏi, bắt AI tự trả lời)
        prompt = f"USER: <image>\n{human_text}\nASSISTANT:"
        
        # 3. Load ảnh và mã hóa Tensor
        img_path = instance['image']
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
            
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        # 4. Generate câu trả lời
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=256, 
                use_cache=True,
                temperature=0.2 # Giữ độ sáng tạo thấp để AI trả lời ổn định
            )
        
        # 5. Giải mã và tách chuỗi
        generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
        pred_gpt_text = generated_text.split("ASSISTANT:")[-1].strip()
        
        pred_choice, pred_exp = parse_llava_response(pred_gpt_text)
        
        # Lưu trữ kết quả
        true_choices.append(true_choice)
        pred_choices.append(pred_choice)
        true_explanations.append(true_exp)
        pred_explanations.append(pred_exp)

    # ==========================================
    # 4. TÍNH TOÁN VÀ IN BÁO CÁO KẾT QUẢ
    # ==========================================
    
    # Lọc bỏ các mẫu mà AI sinh lỗi, không theo định dạng A, B, C, D
    valid_indices = [i for i, (t, p) in enumerate(zip(true_choices, pred_choices)) if p != "UNKNOWN"]
    filtered_true = [true_choices[i] for i in valid_indices]
    filtered_pred = [pred_choices[i] for i in valid_indices]
    
    # Tính Sklearn Metrics
    acc = accuracy_score(filtered_true, filtered_pred) if valid_indices else 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_true, filtered_pred, average='weighted', zero_division=0
    ) if valid_indices else (0, 0, 0, None)
    
    # Tính NLP Metrics (ROUGE & BLEU)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    bleu_scores = []
    smooth_func = SmoothingFunction().method1
    
    for t_exp, p_exp in zip(true_explanations, pred_explanations):
        # ROUGE
        scores = scorer.score(t_exp, p_exp)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # BLEU
        ref_tokens = [nltk.word_tokenize(t_exp.lower())]
        cand_tokens = nltk.word_tokenize(p_exp.lower())
        bleu = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smooth_func)
        bleu_scores.append(bleu)

    print("\n" + "="*60)
    print("📊 BÁO CÁO HIỆU SUẤT MÔ HÌNH (MODEL PERFORMANCE)")
    print("="*60)
    print("1. TÁC VỤ PHÂN LOẠI TRẮC NGHIỆM (A/B/C/D):")
    print(f"   - Tỷ lệ sinh đúng định dạng: {len(valid_indices)}/{num_samples} ({len(valid_indices)/num_samples*100:.1f}%)")
    print(f"   - Accuracy (Độ chính xác):  {acc*100:.2f}%")
    print(f"   - Precision (Độ chuẩn xác): {precision*100:.2f}%")
    print(f"   - Recall (Độ bao phủ):      {recall*100:.2f}%")
    print(f"   - F1-Score:                 {f1*100:.2f}%")
    print("-" * 60)
    print("2. TÁC VỤ SINH VĂN BẢN GIẢI THÍCH (NLP METRICS):")
    print(f"   - ROUGE-1 (Khớp từ đơn):    {np.mean(rouge_scores['rouge1'])*100:.2f}")
    print(f"   - ROUGE-2 (Khớp cụm 2 từ):  {np.mean(rouge_scores['rouge2'])*100:.2f}")
    print(f"   - ROUGE-L (Cấu trúc câu):   {np.mean(rouge_scores['rougeL'])*100:.2f}")
    print(f"   - BLEU Score:               {np.mean(bleu_scores)*100:.2f}")
    print("="*60)

# Thực thi hệ thống đánh giá trên 50 mẫu (Bạn có thể tăng biến num_samples lên nếu máy mạnh)
if __name__ == "__main__":
    evaluate_model(model, processor, eval_dataset, num_samples=50)