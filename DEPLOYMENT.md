# 🚀 AeroRisk Deployment Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION SETUP                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│   📱 Streamlit Cloud          🔗 Render                       │
│   (Dashboard)                 (FastAPI Backend)               │
│   ┌─────────────┐             ┌─────────────┐                │
│   │ dashboard/  │ ─────────▶  │ src/api/    │                │
│   │ app.py      │   HTTP      │ main.py     │                │
│   └─────────────┘             └──────┬──────┘                │
│                                      │                        │
│                               ┌──────▼──────┐                │
│                               │ PostgreSQL  │                │
│                               │ (Optional)  │                │
│                               └─────────────┘                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Step 1: Push to GitHub

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "🚀 AeroRisk v1.0 - ML Aviation Safety Platform"

# Add remote (replace with your repo)
git remote add origin https://github.com/YOUR_USERNAME/aerorisk.git

# Push
git push -u origin main
```

---

## 🔗 Step 2: Deploy Backend to Render

### Option A: Using render.yaml (Automatic)

1. Go to **https://render.com**
2. Click **New** → **Blueprint**
3. Connect your GitHub repo
4. Render will detect `render.yaml` and auto-configure
5. Click **Apply**

### Option B: Manual Setup

1. Go to **https://render.com**
2. Click **New** → **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Name**: `aerorisk-api`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variables:
   - `PYTHON_VERSION` = `3.11`
   - `DATABASE_URL` = Your PostgreSQL URL (optional)
6. Click **Create Web Service**

### After Deployment:
- Note your API URL: `https://aerorisk-api.onrender.com`
- Test: `https://aerorisk-api.onrender.com/health`

---

## 📱 Step 3: Deploy Dashboard to Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **New app**
4. Configure:
   - **Repository**: Your GitHub repo
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
5. **Advanced settings** → Add Secret:
   ```
   AERORISK_API_URL = "https://aerorisk-api.onrender.com"
   ```
6. Click **Deploy!**

### After Deployment:
- Your dashboard: `https://YOUR_APP.streamlit.app`

---

## 🔧 Environment Variables

### Render (Backend)
| Variable | Value | Required |
|----------|-------|----------|
| `PYTHON_VERSION` | `3.11` | Yes |
| `DATABASE_URL` | PostgreSQL URL | Optional |

### Streamlit Cloud (Dashboard)
| Variable | Value | Required |
|----------|-------|----------|
| `AERORISK_API_URL` | Render backend URL | Yes |

---

## 🐛 Troubleshooting

### Backend not starting?
- Check Render logs
- Ensure `requirements.txt` is correct
- Verify Python version

### Dashboard can't connect to API?
- Check `AERORISK_API_URL` secret in Streamlit
- Ensure backend is running (check `/health`)
- Check CORS settings

### Models not loading?
- Models are included in `models/` directory
- Ensure `.gitignore` doesn't exclude `.pkl` files

---

## 📊 Monitoring

### Render Dashboard
- CPU, Memory, Requests
- Logs in real-time

### Streamlit Dashboard
- View usage stats
- Check error logs

---

## 🔒 Security Notes

1. Never commit `.env` files
2. Use Render/Streamlit secrets for sensitive data
3. Enable HTTPS (automatic on both platforms)

---

## 🎉 Done!

Your AeroRisk platform is now live:
- **Dashboard**: `https://YOUR_APP.streamlit.app`
- **API**: `https://aerorisk-api.onrender.com`
- **API Docs**: `https://aerorisk-api.onrender.com/docs`

Share your LinkedIn post with the Streamlit URL! 🚀
