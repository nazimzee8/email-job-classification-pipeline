# Email Job Classification Pipeline

Streamlit app and training pipeline for classifying job-related emails.

## Public app

Production deployment: https://email-classification-pipeline.streamlit.app

## Run locally

From the repository root:

```powershell
.\Scripts\Activate.ps1
streamlit run src\streamlit_app.py
```

If PowerShell blocks activation, run Streamlit directly:

```powershell
.\Scripts\streamlit.exe run src\streamlit_app.py
```

The local app is usually available at `http://localhost:8501`.

## Install dependencies

To recreate the environment from pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Pretrained artifacts

The Streamlit app is configured to load pretrained artifacts from `artifacts\` for fast startup. It does not retrain models during normal app boot.

## Rebuild artifacts

To retrain the full pipeline and overwrite the packaged artifacts:

```powershell
.\Scripts\Activate.ps1
python src\streamlit_app.py --force-retrain
```

Artifacts are written to the `artifacts\` directory.

## Deployment note

For Streamlit Cloud, keep `requirements.txt` pinned and ensure the `artifacts\` directory is committed if you want fast boot without retraining.
