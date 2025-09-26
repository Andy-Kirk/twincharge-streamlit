# Twin Charging Calculator — Full Package

Includes:
- **Air System Library** (Turbo & Supercharger CSV uploaders, model selectors, interpolation, turbo map & SC plot)
- **Engine & Thermal** chain (Turbo → IC1 → SC → IC2) with gauge vs absolute, unit toggles
- **Pump Helper** (presets, CSV upload/paste) + **Extended Pipe Calculator** with graphs and operating points
- **CSV Export** of all key results

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501

## Optional password
Create `.streamlit/secrets.toml` with:
```
APP_PASSWORD = "change-me"
```