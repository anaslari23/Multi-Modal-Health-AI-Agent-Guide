# How to Train Your Medical LLM on Google Colab (Multi-Stage)

I have prepared a Jupyter Notebook (`training_notebook.ipynb`) that implements a **4-Stage Training Pipeline** to create a high-quality medical chatbot.

## The 4 Stages
1.  **Instruction Tuning**: Teaches the model how to chat and follow instructions.
2.  **Domain Adaptation**: Injects medical knowledge from HealthCareMagic and iCliniq.
3.  **Medicine Recommendation**: (Optional) Fine-tunes on MIMIC-IV for safe prescribing.
4.  **Follow-up Questions**: (Optional) Teaches the model to ask clarifying questions.

## Prerequisites
- A Google Account.
- Google Colab (Free tier is okay, but Pro is recommended).
- ~15GB of free space on Google Drive.

## Steps

### 1. Prepare Google Drive
1.  Go to [Google Drive](https://drive.google.com/).
2.  Create a folder named `doctor_online_data`.
3.  Upload the following files to this folder:
    - `training_notebook.ipynb`
    - `cleaned_dataset_with_english_translation.csv` (Stage 1)
    - `HealthCareMagic-100k.json` (Stage 2)
    - `iCliniq.json` (Stage 2)
    - `mimic_iv.csv` (Stage 3 - Optional)
    - `followup_q.json` (Stage 4 - Optional)

> [!NOTE]
> If you don't have the optional files (MIMIC-IV, FollowupQ), the notebook will simply skip those stages and continue.

### 2. Open the Notebook
1.  Right-click `training_notebook.ipynb` in Google Drive.
2.  Select **Open with** > **Google Colaboratory**.

### 3. Run the Training
1.  In Colab, go to **Runtime** > **Change runtime type**.
2.  Select **T4 GPU** (or A100 if you have Pro).
3.  Click **Connect** (top right).
4.  Go to **Runtime** > **Run all**.
5.  You will be asked to authorize Google Drive access. Click **Connect to Google Drive**.

### 4. Wait for Completion
- The notebook will train through each stage sequentially.
- It saves checkpoints in `doctor_online_data/checkpoints/`.
- The final model will be saved as `medical_llama3_final.gguf`.

### 5. Use the Model
- Download `medical_llama3_final.gguf` to your local machine.
- Place it in your project's `models/` directory.
- Update your `.env` to point to this new model.
