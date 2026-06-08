# Final project deep learning (Team Sigma)

Person detection using a custom CNN trained on the LLVIP dataset and a transfer learning approach from a detection algorithm to perfom in our custom generated dataset. The model performs binary classification: person vs no person on infrared night images.

---

## Repository Structure

```
final_project_deep_learning_sigma/
├── dataset_analysis/
│   ├── analysis_dataset_LLVIP_complet.ipynb   # Analysis of the LLVIP dataset
│   └── analysis_my_dataset_complet.ipynb      # Analysis of the custom dataset
├── dataset_generation/
│   └── process_dataset.py                     # Script to generate dataset_labels.csv
├── training_models/
│   ├── handmade_baseline.ipynb                # Handmade baseline model
│   ├── 0_Better_Model.ipynb                   # Experiment 0
│   ├── 1_Better_Model.ipynb                   # Experiment 1
│   ├── 2_Better_Model.ipynb                   # Experiment 2
│   ├── 3_Better_Model.ipynb                   # Experiment 3
│   ├── Best_Model.ipynb                       # Final first best model 
│   └── Test/
│       └── Test_BestModel.ipynb               # Evaluation of the best model on test set
├── results/
│   ├── 0_better_model.pkt                     # Saved weights for experiment 0
│   ├── 1_better_model.pkt                     # Saved weights for experiment 1
│   ├── 2_better_model.pkt                     # Saved weights for experiment 2
│   ├── 3_better_model.pkt                     # Saved weights for experiment 3
│   └── 
├── project_report.pdf
└── project_presentation.pdf
```

---

## Environment

All notebooks are designed to run on Google Colab with a GPU runtime. No local installation is required beyond uploading the repository to Google Drive.

Python dependencies: all pre-installed in Colab or installed automatically by the notebooks


## Setup Instructions

### 1. Clone or download the repository

### 2. Download the data

The data files are **not included** in the repository because they exceed GitHub's size limits.

Download the `data` folder from Google Drive:

> https://drive.google.com/drive/folders/10fUcfnYp_FFE1R7Pi1f7FgZQQKEKj87N?usp=sharing

Once downloaded, extract the folder but do not extract the `.zip` files inside it, the notebooks read from those zip files directly.

### 3. Upload everything to Google Drive

Place both the cloned/downloaded repository folder and the `data` folder inside your Google Drive. The final structure on Drive should look like this:

```
your_drive_folder/
├── Final Project/
│   ├── data/
│   │   └── dataset_LLVIP/
│   │       └── results.zip
│   ├── results/
│   ├── training_models/
│   └── ...
```

---

## Configuring Paths

Each notebook has a **Configuration** cell near the top that defines `project_path`. You must update this variable to match where you placed the project folder in your own Google Drive.

---

## Reproducing the Results

Run the notebooks in the following order:

1. **Dataset analysis** 

2. **Train the experiments**

3. **Evaluate the best model**: test set and reproduce the final metrics

4. **Train the transfern learning model**.

**Note:** Training the best model takes approximately 30–35 minutes on a Colab GPU. If you only want to evaluate, load the pre-saved weights from `results/` and skip the training cell.

---

## Notes

- Make sure to select a GPU runtime in Colab
- Change all teh needed paths in the configuration cells of the notebooks to match your Google Drive structure
- Dowload the data from the link
- The dataset_generation script is used to generate the dataset_labels.csv file required by the training notebooks. However, the generated data is already included in the repository, so running this script is not necessary. If you wish to execute it, you must update the paths to indicate where the resulting dataset should be stored
