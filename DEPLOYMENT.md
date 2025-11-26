# Deploying Your Fine-Tuned Medical Model

Now that you have your `medical_llama3_q4_k_m.gguf` file from Colab, follow these steps to use it in your chatbot.

## 1. Move the Model File
1.  Locate the downloaded `medical_llama3_q4_k_m.gguf` file.
2.  Move it to your project's `models/` directory:
    ```bash
    mv ~/Downloads/medical_llama3_q4_k_m.gguf /Users/anaslari/Desktop/doctor_online/models/
    ```

## 2. Import into Ollama
Ollama needs a `Modelfile` to understand how to run your GGUF.

1.  **Create a Modelfile**:
    Run this command in your terminal to create a file named `Modelfile` in the `models/` directory:
    ```bash
    echo "FROM ./medical_llama3_q4_k_m.gguf
    TEMPLATE \"\"\"{{ if .System }}<|start_header_id|>system<|end_header_id|>

    {{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

    {{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

    {{ .Response }}<|eot_id|>\"\"\"
    PARAMETER stop \"<|start_header_id|>\"
    PARAMETER stop \"<|end_header_id|>\"
    PARAMETER stop \"<|eot_id|>\"" > models/Modelfile
    ```

2.  **Create the Model in Ollama**:
    Run the following command to import the model. This might take a minute.
    ```bash
    ollama create medical-llama3 -f models/Modelfile
    ```

3.  **Verify**:
    Check if the model is listed:
    ```bash
    ollama list
    ```
    You should see `medical-llama3:latest`.

## 3. Configure the Chatbot
Update your backend configuration to use this new model.

1.  Open `mm-hie-backend/.env` (or create it if missing).
2.  Add or update the following line:
    ```env
    MMHIE_REASONER_MODEL=medical-llama3:latest
    ```

## 4. Restart & Test
1.  Restart your backend server:
    ```bash
    # In mm-hie-backend/
    Ctrl+C
    ../.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
2.  Run the benchmark again to see the difference!
    ```bash
    ../.venv/bin/python benchmark_rag.py
    ```
