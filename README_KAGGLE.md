# Exporting Your Model on Kaggle

Since Colab is crashing, we will use Kaggle's robust environment to export your model.

## Prerequisites
- A [Kaggle Account](https://www.kaggle.com/).
- Your `checkpoints` folder (downloaded from Google Drive to your computer).

## Steps

### 1. Create a New Dataset
1.  Go to [Kaggle Datasets](https://www.kaggle.com/datasets).
2.  Click **New Dataset**.
3.  Drag and drop your `checkpoints` folder into the upload area.
4.  Name it `doctor-online-checkpoints`.
5.  Click **Create**.

### 2. Create a New Notebook
1.  Go to [Kaggle Kernels](https://www.kaggle.com/code).
2.  Click **New Notebook**.
3.  **Important**: In the right sidebar, under **Session Options**, set **Accelerator** to **GPU T4 x2** (or P100).

### 3. Add Your Dataset
1.  In the notebook editor, look at the right sidebar.
2.  Click **Add Input**.
3.  Search for `doctor-online-checkpoints` (the dataset you just created).
4.  Click the **+** button to add it.

### 4. Upload the Export Script
1.  In the notebook editor, go to **File** > **Import Notebook**.
2.  Upload the `kaggle_export_notebook.ipynb` file I created for you.

### 5. Run and Download
1.  Click **Run All**.
2.  Wait for the script to finish.
3.  Once done, look at the **Output** section in the right sidebar (you might need to refresh).
4.  You will see `medical_llama3_kaggle.gguf`. Click the **Download** button next to it.

### 6. Deploy
Move the downloaded file to your local `models/` folder and follow the `DEPLOYMENT.md` guide!
