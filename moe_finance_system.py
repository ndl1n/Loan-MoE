import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import asyncio

# Load API Key 
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("[錯誤] 找不到 GEMINI_API_KEY，請確認 .env 檔案設定。")
genai.configure(api_key=API_KEY)

# Encoder 
encoder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def encode_input(text):
    return torch.tensor(encoder.encode([text])).float()

# MoE Gate Model
class MoEGateMulti(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, num_domains=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fke_gate = nn.Linear(hidden_dim, 1)
        self.domain_gate = nn.Linear(hidden_dim, num_domains)

    def forward(self, x):
        x = self.fc(x)
        fke_prob = torch.sigmoid(self.fke_gate(x)).squeeze(-1)
        domain_probs = torch.sigmoid(self.domain_gate(x))
        return fke_prob, domain_probs

# Expert Prompts
EXPERT_PROMPTS = {
    "FKE": "你是金融知識專家，請用簡單明確的方式解釋金融相關知識問題。",
    "LDE": "你是一位專業的中文貸款顧問，請精準、清晰地回答我的問題。若有必要，請以條列式呈現，並提供相關的注意事項或建議。",
    "FRE": "你是一位專業的財務風險評估顧問，請精準、清晰地回答我的問題。若有必要，請以條列式呈現，並提供相關的注意事項或建議。",
    "DTE": "你是一位專業的細節文件顧問，請精準、清晰地回答我的問題。若有必要，請以條列式呈現，並提供相關的注意事項或建議。"
}

EXPERTS = ["LDE", "FRE", "DTE"]

# Call Gemini API
async def gemini_answer(expert, query):
    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        prompt = EXPERT_PROMPTS[expert] + "\n使用者問題：" + query
        response = model.generate_content([prompt])
        return f"[{expert}] {response.text.strip()}"
    except Exception as e:
        return f"[{expert}] [錯誤] Gemini 回應失敗：{str(e)}"

# Inference
def run_moe_inference(user_input, model, threshold=0.5, top_k=2):
    x = encode_input(user_input)
    with torch.no_grad():
        fke_prob, domain_probs = model(x)

    use_fke = fke_prob.item() > threshold
    top_indices = domain_probs[0].topk(top_k).indices.tolist()
    selected_experts = [EXPERTS[i] for i in top_indices]

    if use_fke:
        selected_experts.insert(0, "FKE")

    async def gather_responses():
        tasks = [gemini_answer(expert, user_input) for expert in selected_experts]
        return await asyncio.gather(*tasks)

    return asyncio.run(gather_responses())

# Main for CLI Test
if __name__ == "__main__":
    model = MoEGateMulti()
    user_input = input("請輸入問題：")
    responses = run_moe_inference(user_input, model)
    print("\n[系統回應]")
    for r in responses:
        print(r)
