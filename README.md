# Twin Charging Calculator — Full Package (Username + Password)

Now includes a username **and** password login using Streamlit Secrets.

## Set credentials (Streamlit Cloud)
- App → **⋯ → Settings → Secrets**:
```
APP_USER = "yourUser"
APP_PASSWORD = "change-me"
```
If either secret is missing, the app runs **without** a login (handy for local dev).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – app code (with auth gate)
- `requirements.txt`
- `.streamlit/secrets.toml.example` – local dev example