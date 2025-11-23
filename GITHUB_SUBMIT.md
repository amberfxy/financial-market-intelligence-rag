# GitHub 提交指南

## 📋 提交步骤

### 步骤 1: 检查当前状态

```bash
# 查看当前Git状态
git status

# 查看远程仓库（如果已配置）
git remote -v
```

### 步骤 2: 初始化Git仓库（如果还没有）

```bash
# 如果还没有初始化Git仓库
git init
```

### 步骤 3: 添加所有文件

```bash
# 添加所有应该提交的文件（.gitignore会自动排除不需要的文件）
git add .

# 检查将要提交的文件
git status
```

### 步骤 4: 提交更改

```bash
# 提交所有更改
git commit -m "Initial commit: Financial Market Intelligence RAG System

- Complete RAG pipeline implementation
- Streamlit UI for querying
- FAISS vector store integration
- Local LLM (Mistral 7B) support
- Docker configuration
- Colab notebook for cloud deployment"
```

### 步骤 5: 添加远程仓库

```bash
# 添加GitHub远程仓库
git remote add origin https://github.com/amberfxy/financial-market-intelligence-rag.git

# 或者如果已经存在，更新URL
git remote set-url origin https://github.com/amberfxy/financial-market-intelligence-rag.git
```

### 步骤 6: 推送到GitHub

```bash
# 推送到main分支
git branch -M main
git push -u origin main
```

---

## 🔐 认证方式

### 方式 1: Personal Access Token (推荐)

1. 访问 https://github.com/settings/tokens
2. 生成新的 token (classic)
3. 选择权限：`repo` (完整仓库访问)
4. 复制 token
5. 推送时使用：
   ```bash
   git push -u origin main
   # Username: 你的GitHub用户名
   # Password: 粘贴你的token（不是密码）
   ```

### 方式 2: SSH (如果已配置)

```bash
# 使用SSH URL
git remote set-url origin git@github.com:amberfxy/financial-market-intelligence-rag.git
git push -u origin main
```

### 方式 3: GitHub Desktop

1. 打开 GitHub Desktop
2. File → Add Local Repository
3. 选择项目目录
4. 点击 Publish repository

---

## ⚠️ 提交前检查清单

### ✅ 确保以下文件已包含：

- [x] 所有源代码 (`src/`, `ui/`, `scripts/`)
- [x] 配置文件 (`requirements.txt`, `Dockerfile`, `docker-compose.yml`)
- [x] 文档 (`README.md`, `models/README.md`)
- [x] Colab notebook (`Financial_Market_RAG_Colab.ipynb`)
- [x] `.gitignore` 文件

### ❌ 确保以下文件被排除：

- [ ] 数据文件 (`data/raw/*`, `data/processed/*`)
- [ ] 模型文件 (`models/*.gguf`)
- [ ] 索引文件 (`vectorstore/*`)
- [ ] 凭证文件 (`kaggle.json`, `.env`)
- [ ] 缓存文件 (`__pycache__/`, `.venv/`)

---

## 🔍 验证提交

### 检查文件大小

```bash
# 检查是否有大文件（>100MB）
find . -type f -size +50M ! -path "./.git/*" ! -path "./.venv/*"
```

### 检查将要提交的文件

```bash
# 查看将要提交的文件列表
git ls-files

# 查看文件数量
git ls-files | wc -l
```

---

## 🚀 快速命令总结

```bash
# 1. 初始化（如果需要）
git init

# 2. 添加文件
git add .

# 3. 提交
git commit -m "Initial commit: Financial Market Intelligence RAG System"

# 4. 添加远程仓库
git remote add origin https://github.com/amberfxy/financial-market-intelligence-rag.git

# 5. 推送到GitHub
git branch -M main
git push -u origin main
```

---

## 🔧 故障排除

### 问题 1: 认证失败

**错误**: `fatal: could not read Username`

**解决**:
- 使用 Personal Access Token 而不是密码
- 或配置 SSH 密钥

### 问题 2: 大文件警告

**错误**: 文件太大无法推送

**解决**:
- 确保 `.gitignore` 正确配置
- 大文件（模型、数据）不应提交
- 使用 Git LFS（如果需要）

### 问题 3: 远程仓库已存在

**错误**: `remote origin already exists`

**解决**:
```bash
# 更新远程URL
git remote set-url origin https://github.com/amberfxy/financial-market-intelligence-rag.git
```

### 问题 4: 分支名称

**错误**: 分支名称不匹配

**解决**:
```bash
# 重命名分支为main
git branch -M main
```

---

## 📝 提交信息建议

### 首次提交

```
Initial commit: Financial Market Intelligence RAG System

- Complete RAG pipeline with BGE embeddings and FAISS
- Streamlit UI for interactive querying
- Local LLM (Mistral 7B) integration
- Docker configuration for deployment
- Colab notebook for cloud execution
- Comprehensive documentation
```

### 后续提交

```
Update: [描述更改内容]

- [具体更改1]
- [具体更改2]
```

---

## ✅ 提交后验证

1. 访问 https://github.com/amberfxy/financial-market-intelligence-rag
2. 确认所有文件都已上传
3. 检查 README.md 是否正确显示
4. 验证 `.gitignore` 是否正确排除了大文件

---

## 💡 提示

- **首次推送**: 可能需要几分钟，取决于文件数量
- **后续更新**: 使用 `git add .` → `git commit -m "message"` → `git push`
- **查看历史**: `git log` 查看提交历史
- **撤销更改**: `git reset HEAD~1` 撤销最后一次提交（保留文件）

