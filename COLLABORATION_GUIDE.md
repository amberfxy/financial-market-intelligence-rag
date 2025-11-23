# Collaboration Guide - GitHub vs Colab

## Overview

**GitHub** is for collaborative development (version control, code review, task tracking).  
**Colab** is only for running and demonstration (free GPU, no local setup).

---

## Recommended Workflow

```
Development (GitHub)
    ↓
Code Review & Merge
    ↓
Push to GitHub
    ↓
Demo (Colab)
    ↓
Clone from GitHub to Colab
```

---

## Collaboration Steps

### Step 1: Invite Teammate

1. Go to: https://github.com/amberfxy/financial-market-intelligence-rag/settings/access
2. Click "Add people"
3. Enter teammate's GitHub username or email
4. Set permission: **Write**

### Step 2: Teammate Setup

```bash
# 1. Clone repository
git clone https://github.com/amberfxy/financial-market-intelligence-rag.git
cd financial-market-intelligence-rag

# 2. Create own branch
git checkout -b soonbee-dev

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start development
# Edit code...

# 5. Commit changes
git add .
git commit -m "Add: feature description"
git push origin soonbee-dev
```

### Step 3: Code Review

1. **Create Pull Request** on GitHub
   - Select: `soonbee-dev` → `main`
   - Describe changes

2. **Review Code**
   - Check changes
   - Add comments
   - Approve or request changes

3. **Merge**
   - Click "Merge pull request"
   - Code merged to main

4. **Update Local Code**
   ```bash
   git checkout main
   git pull origin main
   ```

---

## Branch Strategy

```
main (production)
  ├── soonbee-dev (teammate's branch)
  ├── amber-dev (your branch)
  └── feature/* (feature branches)
```

---

## Best Practices

### Commit Messages

```bash
# Good
git commit -m "Add: FAISS vector store implementation"
git commit -m "Fix: Embedding batch size optimization"

# Avoid
git commit -m "update"
git commit -m "fix bug"
```

### Daily Workflow

```bash
# Start of day
git checkout main
git pull origin main

# End of day
git add .
git commit -m "Daily progress: ..."
git push origin your-branch
```

### Handle Conflicts

```bash
# 1. Pull latest code
git checkout main
git pull origin main

# 2. Merge to your branch
git checkout your-branch
git merge main

# 3. Resolve conflicts (edit files)
# 4. Commit resolved code
git add .
git commit -m "Resolve merge conflicts"
git push origin your-branch
```

---

## File Management

### ✅ Upload to GitHub

- All source code (`src/`, `ui/`, `scripts/`)
- Configuration files (`requirements.txt`, `Dockerfile`, `.gitignore`)
- Documentation (`README.md`, `PROJECT_GUIDE.md`, `final-report.tex`)
- Colab notebook (`Financial_Market_RAG_Colab.ipynb`)

### ❌ Do NOT Upload

- Large files: `data/raw/*.csv`, `models/*.gguf`, `vectorstore/*.faiss`
- Sensitive info: `kaggle.json`, `.env`
- Temporary files: `__pycache__/`, `*.log`

---

## Communication Tools

- **GitHub Issues**: Task tracking
- **GitHub Discussions**: Technical discussions
- **Pull Request Comments**: Code review discussions
- **Instant Messaging**: Daily communication

---

## Summary

1. **Development**: Use GitHub (branches, PRs, code review)
2. **Demo**: Use Colab (clone from GitHub, run with GPU)
3. **Communication**: GitHub Issues + instant messaging

### Don't Do

❌ Don't collaborate in Colab  
❌ Don't commit directly to main (use branches)  
❌ Don't commit large files  
❌ Don't commit sensitive information

---

**Remember: GitHub for collaboration, Colab for demonstration!** 🚀
