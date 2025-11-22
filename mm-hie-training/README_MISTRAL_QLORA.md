# Mistral-7B QLoRA Medical Reasoning Fine-Tune

Steps to run:

1. Preprocess:
   python data_preprocessing/preprocess_medqa.py
   python data_preprocessing/preprocess_medmcqa.py
   python data_preprocessing/merge_datasets.py

2. Train:
   accelerate launch train/qlora_train_mistral.py

3. Evaluate:
   python eval/eval_mistral_finetuned.py

4. Inference:
   python inference/run_inference.py
