# Email Job Classification Pipeline

Small Streamlit app and training pipeline for classifying job-related emails.

## Run the app

From the repository root:

```powershell
.\Scripts\Activate.ps1
streamlit run src\streamlit_app.py
```

If PowerShell blocks activation, run Streamlit directly:

```powershell
.\Scripts\streamlit.exe run src\streamlit_app.py
```

The app will usually be available at `http://localhost:8501`.

## Refresh artifacts

To retrain the full pipeline and overwrite the packaged artifacts:

```powershell
.\Scripts\Activate.ps1
python src\streamlit_app.py --force-retrain
```

Artifacts are written to the `artifacts\` directory.

## Reinstall dependencies

To recreate the environment from pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
