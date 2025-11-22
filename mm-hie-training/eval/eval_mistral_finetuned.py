from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def evaluate(model_path, questions):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    for q in questions:
        prompt = f"<s>[INST] {q} [/INST]"
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        print("Q:", q)
        print("A:", tok.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    evaluate("checkpoints/mistral-medical-qlora", [
        "A 25 year old male has fever and cough for 3 days. What is the likely diagnosis?"
    ])
