import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# LoRA 模型路徑
LORA_MODELS = {
    "LDE": "lde_lora_model",
    "FRE": "fre_lora_model",
    "DTE": "dte_lora_model"
}

# 專家提示詞
EXPERT_PROMPTS = {
    "LDE": "你是一位專業的中文貸款顧問，請精準、清晰地回答我的問題。",
    "FRE": "你是一位專業的財務風險評估顧問，請精準、清晰地回答我的問題。",
    "DTE": "你是一位專業的細節文件顧問，請精準、清晰地回答我的問題。"
}

def load_lora_model(expert, path):
    """載入指定 LoRA 模型並建立 pipeline"""
    if not os.path.exists(path):
        print(f"[錯誤] {expert} 的 LoRA 模型路徑不存在: {path}")
        return None
    
    print(f"[系統] 載入 {expert} 的 LoRA 模型：{path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",      # 自動選 GPU
        torch_dtype=torch.float16,
        load_in_4bit=True       # 開啟 4-bit
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def test_expert(expert, generator, query="請幫我解釋房貸申請需要準備哪些文件？"):
    """測試單一專家的 LoRA 模型輸出"""
    if generator is None:
        return f"[{expert}] 模型未載入"
    prompt = EXPERT_PROMPTS[expert] + "\n使用者問題：" + query
    output = generator(prompt, max_new_tokens=200, do_sample=True, top_p=0.9)
    return f"[{expert}] {output[0]['generated_text']}"

if __name__ == "__main__":
    # 測試每個專家
    for expert, path in LORA_MODELS.items():
        gen = load_lora_model(expert, path)
        response = test_expert(expert, gen)
        print("\n==============================")
        print(response)
        print("==============================\n")
