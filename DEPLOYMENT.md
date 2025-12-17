# GrantScout Deployment Guide

## Quick Deployment to Streamlit Community Cloud (FREE)

Follow these steps to deploy your GrantScout app and share it with your professor.

---

## ✅ Pre-Deployment Checklist (COMPLETED)

The following files have been prepared for deployment:

- ✅ `packages.txt` - Tells Streamlit Cloud to install Tesseract OCR
- ✅ `requirements.txt` - Updated to use CPU-only PyTorch (lighter weight)
- ✅ `.gitignore` - Configured to exclude `.env` file (protects API keys)
- ✅ `src/data/` - Researcher database ready to be committed

---

## Step 1: Push Code to GitHub

### 1.1 Add all deployment files to Git

```bash
git add packages.txt requirements.txt .gitignore src/data/
```

### 1.2 Commit the changes

```bash
git commit -m "Prepare app for Streamlit Cloud deployment

- Add packages.txt for Tesseract OCR dependency
- Update requirements.txt to use CPU-only PyTorch
- Fix .gitignore to include src/data/ while protecting .env
- Include researcher database for deployment"
```

### 1.3 Push to GitHub

```bash
git push origin grant-coach
```

**⚠️ IMPORTANT:** Your `.env` file will NOT be pushed (it's in .gitignore). This is correct for security!

---

## Step 2: Deploy on Streamlit Community Cloud

### 2.1 Go to Streamlit Cloud
Visit: [https://share.streamlit.io/](https://share.streamlit.io/)

### 2.2 Sign in with GitHub
Click "Sign in with GitHub" and authorize Streamlit.

### 2.3 Create New App
1. Click **"New app"** button
2. Select your repository and branch:
   - **Repository:** `your-username/GrantScout`
   - **Branch:** `grant-coach` (or `main` if you merged)
   - **Main file path:** `main.py`
3. Click **"Deploy!"**

---

## Step 3: Add Your API Key (CRITICAL)

Your app will start deploying but will crash because it can't find your OpenAI API key. Here's how to add it securely:

### 3.1 Open App Settings
1. Wait for the deployment to start (it will show errors - that's expected)
2. Click **"⋮" (three dots)** in the bottom right corner
3. Select **"Settings"**

### 3.2 Add Secrets
1. In Settings, click the **"Secrets"** tab
2. Paste the following (update with your actual API key):

```toml
OPENAI_API_KEY = "sk-proj-your-actual-key-here"
OPENAI_MODEL = "gpt-4o-mini"
```

3. Click **"Save"**
4. The app will automatically restart with your secrets

---

## Step 4: Share Your App

Once deployment completes successfully:
1. Your app will be live at: `https://your-app-name.streamlit.app`
2. Copy this URL and share it with your professor
3. Anyone with the link can use the app!

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"
- Check that all dependencies are in `requirements.txt`
- Look at the deployment logs for which package is missing

### "Tesseract not found"
- Verify `packages.txt` is in your repository root
- It should contain just one line: `tesseract-ocr`

### "Out of memory" or deployment crashes
- The CPU-only PyTorch should help
- If still failing, consider removing heavy dependencies or using a different host

### ".env file not found" errors
- This is expected! Add your secrets via Streamlit Settings (Step 3)
- Never commit your .env file to GitHub

### Large file warnings from Git
- Your largest file (`tfidf_model.pkl`) is 87MB
- This is under GitHub's 100MB limit
- If you get warnings, it should still work

---

## 📊 What Gets Deployed

**Included in deployment:**
- All Python code (`main.py`, `src/`)
- Researcher database (`src/data/researcher_db/`)
- Dependencies (via `requirements.txt`)
- System packages (via `packages.txt`)

**NOT included (protected by .gitignore):**
- Your `.env` file (API keys)
- Root `/data/` folder (user uploads)
- Root `/temp_uploads/` folder
- Virtual environment
- Cache files

---

## 🚀 Next Steps After Deployment

1. Test all features in the deployed app
2. Share the URL with your professor
3. Monitor usage via Streamlit Cloud dashboard
4. Update by pushing new commits to GitHub (auto-redeploys)

---

## Need Help?

- Streamlit Docs: https://docs.streamlit.io/deploy
- Streamlit Community: https://discuss.streamlit.io/
- Check deployment logs in the Streamlit Cloud dashboard for errors
