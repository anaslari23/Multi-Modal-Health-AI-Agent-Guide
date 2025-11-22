"""
Fine-tuned Medical LLM wrapper for medicine suggestions.
Uses the LoRA-adapted TinyLlama model trained on medicine dataset.
"""

from pathlib import Path
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class FineTunedMedicalLLM:
    """Wrapper for the fine-tuned medical LLM."""
    
    def __init__(
        self,
        base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_path: str = "./models/medical-medicine-lora"
    ):
        self.base_model_name = base_model_name
        self.lora_path = Path(lora_path)
        self._model = None
        self._tokenizer = None
        
    def _ensure_loaded(self):
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return
            
        print(f"Loading fine-tuned medical model from {self.lora_path}...")
        
        print(f"Loading tokenizer from {self.lora_path}...")
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.lora_path))
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"
        
        # Load base model
        # Use explicit CPU loading to avoid "auto" issues on Mac
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        print(f"Loading base model {self.base_model_name} with device_map={device_map}...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=None, # Let transformers handle it
            low_cpu_mem_usage=True
        )
        print("Base model loaded.")
        
        # Load LoRA adapters
        print(f"Loading LoRA adapters from {self.lora_path}...")
        self._model = PeftModel.from_pretrained(base_model, str(self.lora_path))
        self._model.eval()  # Set to evaluation mode
        
        print("✅ Fine-tuned medical model loaded successfully!")
        
    def generate(
        self,
        instruction: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response for the given instruction.
        
        Args:
            instruction: The medical query/instruction
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response text
        """
        self._ensure_loaded()
        
        # Format as instruction-following prompt
        prompt = f"""### Instruction:
{instruction}

### Response:
"""
        
        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        # Decode
        full_response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part (after "### Response:")
        if "### Response:" in full_response:
            response = full_response.split("### Response:")[-1].strip()
        else:
            response = full_response.strip()
            
        return response
    
    def medicine_suggestion(
        self,
        disease: str,
        patient_info: Optional[str] = None
    ) -> str:
        """
        Suggest medicines for a given disease.
        
        Args:
            disease: The disease/condition
            patient_info: Optional patient information
            
        Returns:
            Medicine suggestion with safety information
        """
        instruction = f"What medicine can be used for {disease}?"
        if patient_info:
            instruction += f" Patient info: {patient_info}"
            
        return self.generate(instruction)
    
    def medicine_info(self, medicine_name: str) -> str:
        """Get information about a specific medicine."""
        instruction = f"Tell me about {medicine_name}"
        return self.generate(instruction)
    
    def contraindications(self, medicine_name: str) -> str:
        """Get contraindications for a medicine."""
        instruction = f"What are the contraindications for {medicine_name}?"
        return self.generate(instruction)


# Singleton instance (lazy loaded)
_medical_llm_instance: Optional[FineTunedMedicalLLM] = None

def get_medical_llm() -> FineTunedMedicalLLM:
    """Get or create the fine-tuned medical LLM instance."""
    global _medical_llm_instance
    if _medical_llm_instance is None:
        _medical_llm_instance = FineTunedMedicalLLM()
    return _medical_llm_instance
