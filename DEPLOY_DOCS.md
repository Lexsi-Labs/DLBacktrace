# Deploying Documentation to GitHub Pages

This guide explains how to deploy the DL-Backtrace documentation to GitHub Pages.

---

## Quick Start

### Option 1: Automatic Deployment (Recommended)

The documentation automatically deploys to GitHub Pages when you:

1. **Push changes to `main` branch** that affect:
   - Files in `docs/` directory
   - `mkdocs.yml` configuration
   - `.github/workflows/docs.yml` workflow

2. **Manually trigger deployment**:
   - Go to: https://github.com/AryaXAI/DL-Backtrace/actions
   - Select "Deploy MkDocs to GitHub Pages" workflow
   - Click "Run workflow" button
   - Select branch and click "Run workflow"

### Option 2: Manual Deployment (Local)

Deploy directly from your local machine:

```bash
# 1. Install dependencies
pip install -r docs-requirements.txt

# 2. Test locally (optional)
mkdocs serve
# Visit http://localhost:8000

# 3. Deploy to GitHub Pages
mkdocs gh-deploy --force --clean
```

---

## Setup Requirements

### 1. Enable GitHub Pages

Ensure GitHub Pages is enabled in your repository:

1. Go to: https://github.com/AryaXAI/DL-Backtrace/settings/pages
2. Under "Build and deployment":
   - **Source**: Deploy from a branch
   - **Branch**: `gh-pages` / `/ (root)`
3. Click "Save"

### 2. Verify Workflow Permissions

Ensure GitHub Actions has write permissions:

1. Go to: https://github.com/AryaXAI/DL-Backtrace/settings/actions
2. Scroll to "Workflow permissions"
3. Select "Read and write permissions"
4. Check "Allow GitHub Actions to create and approve pull requests"
5. Click "Save"

---

## Documentation URL

After successful deployment, your documentation will be available at:

**https://aryaxai.github.io/DL-Backtrace/**

---

## Workflow Details

### Automatic Triggers

The workflow (`.github/workflows/docs.yml`) runs when:

- Changes are pushed to `main` branch affecting documentation files
- Manually triggered via GitHub Actions UI

### Build Process

1. **Checkout**: Fetches repository code
2. **Setup Python**: Installs Python 3.10
3. **Install Dependencies**: Installs MkDocs and plugins from `docs-requirements.txt`
4. **Configure Git**: Sets up git for deployment
5. **Build & Deploy**: Runs `mkdocs gh-deploy` to build and push to `gh-pages` branch

### Deployment Time

- **Build**: ~2-3 minutes
- **GitHub Pages update**: ~1-2 minutes after build
- **Total**: ~5 minutes from push to live

---

## Troubleshooting

### Issue 1: Workflow Not Running

**Problem**: Changes pushed but workflow doesn't trigger

**Solutions**:
- Ensure changes are in `docs/`, `mkdocs.yml`, or `.github/workflows/docs.yml`
- Check you're pushing to `main` branch
- Verify workflow file syntax is correct
- Check Actions tab for any disabled workflows

### Issue 2: Build Fails

**Problem**: Workflow runs but build fails

**Solutions**:

1. **Check error logs**:
   - Go to Actions tab
   - Click on failed workflow run
   - Expand failed step to see error

2. **Common errors**:
   ```
   Error: Config file 'mkdocs.yml' does not exist
   ```
   **Fix**: Ensure `mkdocs.yml` is in repository root

   ```
   Error: The command mkdocs does not exist
   ```
   **Fix**: Check `docs-requirements.txt` includes `mkdocs>=1.5.0`

   ```
   Error: Documentation file 'xyz.md' not found
   ```
   **Fix**: Check file path in `mkdocs.yml` nav section matches actual file location

3. **Test locally first**:
   ```bash
   mkdocs build --strict
   ```
   This shows all errors before deployment

### Issue 3: Pages Not Updating

**Problem**: Workflow succeeds but site doesn't update

**Solutions**:

1. **Check GitHub Pages settings**:
   - Settings → Pages → Source should be `gh-pages` branch

2. **Clear browser cache**:
   - Hard refresh: Ctrl+F5 (Windows/Linux) or Cmd+Shift+R (Mac)
   - Or use incognito/private mode

3. **Check deployment status**:
   - Go to: https://github.com/AryaXAI/DL-Backtrace/deployments
   - Look for recent `github-pages` deployment

4. **Wait longer**:
   - GitHub Pages can take 1-2 minutes to update after deployment

### Issue 4: 404 Not Found

**Problem**: Site loads but pages show 404

**Solutions**:

1. **Check `site_url` in `mkdocs.yml`**:
   ```yaml
   site_url: https://aryaxai.github.io/DL-Backtrace/
   ```
   Must match your GitHub Pages URL

2. **Check file paths in navigation**:
   - Paths in `nav` section must match actual file locations
   - Use forward slashes: `guide/pytorch/overview.md`
   - Files must have `.md` extension

3. **Verify files exist**:
   ```bash
   ls -R docs/
   ```

### Issue 5: Missing Assets/Images

**Problem**: Images or CSS/JS files not loading

**Solutions**:

1. **Check asset paths**:
   ```markdown
   # Correct - relative to current file
   ![Logo](../assets/images/logo.png)
   
   # Or relative to docs root
   ![Logo](assets/images/logo.png)
   ```

2. **Verify assets directory**:
   ```
   docs/
   ├── assets/
   │   ├── images/
   │   │   └── logo.png
   │   └── stylesheets/
   ```

3. **Check `mkdocs.yml` extra files**:
   ```yaml
   extra_javascript:
     - javascripts/custom.js
   extra_css:
     - stylesheets/custom.css
   ```

### Issue 6: Permission Denied

**Problem**: `mkdocs gh-deploy` fails with permission error

**Solutions**:

1. **For GitHub Actions**:
   - Ensure workflow has `permissions: contents: write`
   - Check repository Actions permissions (Settings → Actions → Workflow permissions)

2. **For local deployment**:
   - Ensure you have push access to repository
   - Check git credentials: `git remote -v`
   - Try: `git config --global credential.helper store`

---

## Development Workflow

### 1. Edit Documentation Locally

```bash
# Navigate to repository
cd DL-Backtrace

# Install dependencies (first time only)
pip install -r docs-requirements.txt

# Start live preview server
mkdocs serve

# Open browser to http://localhost:8000
# Edit files in docs/ - changes auto-reload
```

### 2. Test Build

```bash
# Build documentation (outputs to site/ directory)
mkdocs build

# Build with strict mode (fails on warnings)
mkdocs build --strict

# Clean build
mkdocs build --clean
```

### 3. Preview Before Deploy

```bash
# Serve built site
cd site
python -m http.server 8000

# Or use mkdocs serve
mkdocs serve
```

### 4. Deploy

**Option A: Push to main (automatic)**
```bash
git add docs/ mkdocs.yml
git commit -m "docs: update documentation"
git push origin main
```

**Option B: Manual deployment**
```bash
mkdocs gh-deploy --force --clean
```

---

## Configuration Files

### mkdocs.yml

Main configuration file at repository root:

```yaml
site_name: DL-Backtrace Documentation
site_url: https://aryaxai.github.io/DL-Backtrace/
repo_url: https://github.com/aryaxai/DL-Backtrace

theme:
  name: material
  # ... theme settings

nav:
  - Home: index.md
  # ... navigation structure
```

### docs-requirements.txt

Python dependencies for building docs:

```
mkdocs>=1.5.0
mkdocs-material>=9.5.0
pymdown-extensions>=10.7
mkdocs-minify-plugin>=0.8.0
mkdocs-git-revision-date-localized-plugin>=1.2.0
```

### .github/workflows/docs.yml

GitHub Actions workflow for automated deployment.

---

## Best Practices

1. **Always test locally before deploying**:
   ```bash
   mkdocs serve
   ```

2. **Use strict mode to catch errors**:
   ```bash
   mkdocs build --strict
   ```

3. **Check navigation structure**:
   - All files in `nav` must exist
   - Paths must be relative to `docs/` directory

4. **Optimize images**:
   - Use compressed images (PNG, JPEG, WebP)
   - Keep images under 500KB when possible
   - Use appropriate dimensions

5. **Link checking**:
   ```bash
   # Install linkchecker
   pip install linkchecker
   
   # Build and check
   mkdocs build
   linkchecker site/
   ```

6. **Version your changes**:
   - Commit documentation changes separately
   - Use descriptive commit messages: `docs: add CUDA guide`

---

## Monitoring

### Check Deployment Status

1. **GitHub Actions**:
   - https://github.com/AryaXAI/DL-Backtrace/actions

2. **Deployments**:
   - https://github.com/AryaXAI/DL-Backtrace/deployments

3. **Build logs**:
   - Click on workflow run → Click on job → Expand steps

### Analytics (Optional)

Add Google Analytics to track documentation usage:

```yaml
# In mkdocs.yml
extra:
  analytics:
    provider: google
    property: G-XXXXXXXXXX
```

---

## Additional Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

## Support

If you encounter issues not covered here:

1. Check GitHub Actions logs for detailed error messages
2. Search existing issues: https://github.com/AryaXAI/DL-Backtrace/issues
3. Create new issue with:
   - Error message
   - Workflow run link
   - Steps to reproduce
   - Screenshots if applicable

---

**Last Updated**: October 24, 2025

