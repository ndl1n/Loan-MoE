from moe_model import load_gate_model, run_moe_inference

if __name__ == "__main__":
    model = load_gate_model()
    user_input = input("請輸入問題：")
    responses = run_moe_inference(user_input, model)
    print("\n[系統回應]")
    for r in responses:
        print(r)