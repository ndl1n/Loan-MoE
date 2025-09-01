import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ======================
# 初始化 & 設定
# ======================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("[錯誤] 找不到 GEMINI_API_KEY，請確認 .env 檔案設定。")
genai.configure(api_key=API_KEY)

# Sentence Encoder
encoder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def encode_input(text):
    """將文字轉換為向量張量"""
    return torch.tensor(encoder.encode([text])).float()


# ======================
# MoE Gate Model 定義
# ======================
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


# ======================
# Expert 設定
# ======================
EXPERT_PROMPTS = {
    "FKE": "你是金融知識專家，請用簡單明確的方式解釋金融相關知識問題。",
    "LDE": "你是一位專業的中文貸款顧問，請精準、清晰地回答我的問題。",
    "FRE": "你是一位專業的財務風險評估顧問，請精準、清晰地回答我的問題。",
    "DTE": "你是一位專業的細節文件顧問，請精準、清晰地回答我的問題。"
}
EXPERTS = ["LDE", "FRE", "DTE"]

# ======================
# LoRA 模型載入 (替換成你的路徑)
# ======================
LORA_MODELS = {
    "LDE": "lde_lora_model",
    "FRE": "fre_lora_model",
    "DTE": "dte_lora_model"
}

lora_pipelines = {}
for expert, path in LORA_MODELS.items():
    if os.path.exists(path):
        print(f"[系統] 載入 {expert} 的 LoRA 模型：{path}")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",      # 自動選 GPU
            torch_dtype=torch.float16,
            load_in_4bit=True       # 開啟 4-bit
        )
        lora_pipelines[expert] = pipeline("text-generation", model=model, tokenizer=tokenizer)
    else:
        print(f"[警告] {expert} 的 LoRA 模型路徑不存在：{path}")


# ======================
# Gemini API 呼叫
# ======================
async def gemini_answer(expert, query):
    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        prompt = EXPERT_PROMPTS[expert] + "\n使用者問題：" + query
        response = model.generate_content([prompt])
        return f"[{expert}] {response.text.strip()}"
    except Exception as e:
        return f"[{expert}] [錯誤] Gemini 回應失敗：{str(e)}"


# ======================
# LoRA 回應
# ======================
def lora_answer(expert, query, max_new_tokens=200):
    if expert not in lora_pipelines:
        return f"[{expert}] [錯誤] 尚未載入 LoRA 模型"
    
    generator = lora_pipelines[expert]
    full_prompt = EXPERT_PROMPTS[expert] + "\n使用者問題：" + query
    outputs = generator(full_prompt, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9)
    return f"[{expert}] {outputs[0]['generated_text']}"


# ======================
# 推理流程
# ======================
def run_moe_inference(user_input, model, threshold=0.5, top_k=1, verbose=True):
    x = encode_input(user_input)
    with torch.no_grad():
        fke_prob, domain_probs = model(x)

    # 印出機率
    if verbose:
        print(f"\n[FKE 機率] {fke_prob.item():.4f}")
        for i, expert in enumerate(EXPERTS):
            print(f"[{expert} 機率] {domain_probs[0][i].item():.4f}")

    # 判斷選用專家
    all_probs = [fke_prob.item()] + domain_probs[0].tolist()
    max_prob = max(all_probs)
    fke_is_max = fke_prob.item() == max_prob

    # 防呆條件：如果四個模型機率都低於 threshold
    if max_prob < threshold and fke_is_max:
        selected_experts = ["FKE"]
    else:
        use_fke = fke_prob.item() > threshold
        top_indices = domain_probs[0].topk(top_k).indices.tolist()
        selected_experts = [EXPERTS[i] for i in top_indices]
        if use_fke:
            selected_experts.insert(0, "FKE")

    # === 分流回應 ===
    async def gather_responses():
        tasks = []
        for expert in selected_experts:
            if expert == "FKE":
                tasks.append(gemini_answer(expert, user_input))
            else:
                # LoRA 是同步函數，所以直接執行
                tasks.append(asyncio.to_thread(lora_answer, expert, user_input))
        return await asyncio.gather(*tasks)

    return asyncio.run(gather_responses())


# ======================
# 載入/建立模型
# ======================
def load_gate_model(path="moe_model_path\moe_gate_model-384.pth"):
    model = MoEGateMulti()
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        print(f"[系統] 成功載入已訓練模型：{path}")
    else:
        print("[警告] 找不到已保存的模型，將使用隨機初始化的模型。")
    model.eval()
    return model
