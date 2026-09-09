# Brain Tumor MRI EDA


## Contact
- Bhavin Yedlapalli
- Senior Biomedical Engineering 
- School: yedla008@umn.edu
- Personal: bhavinyed05@gmail.com

Exploratory data analysis for brain tumor MRI images. The project extracts image metadata and creates a dashboard with sample images, intensity distributions, image dimensions, and class counts.

## Requirements
- Python 3.9+
- The packages listed in `requirements.txt`
- The MRI dataset arranged under `brain_tumor_data/Training/`

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The application expects these class folders:

```text
brain_tumor_data/
└── Training/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The dataset can be obtained from the, Brain Tumor MRI Dataset on Kaggle, (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

## Run

From the project root, run:

```bash
python run.py
```

The generated dashboard is saved as `Brain EDA.png`.

## Project Structure

```text
run.py                  # Application entry point
src/data_pipeline.py    # Image metadata and Kaggle dataset helpers
src/dashboard.py        # Dashboard visualizations
requirements.txt        # Python dependencies
```

## Note

This project is intended for exploratory analysis and educational use. It is not a medical diagnostic tool.
