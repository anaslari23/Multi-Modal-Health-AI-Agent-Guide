from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class MedicalLLM:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").eval()

    def ask(self, text):
        prompt = f"<s>[INST] {text} [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=300)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


if __name__ == "__main__":
    llm = MedicalLLM("checkpoints/mistral-medical-qlora")
    print(llm.ask("Patient with chest pain and dyspnea. What should we suspect?"))
