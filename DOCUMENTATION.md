# DL-Backtrace Documentation

Comprehensive documentation for the DL-Backtrace explainable AI framework.

---

## 📚 Documentation Structure

The documentation is built with **MkDocs** and **Material for MkDocs** theme.

### Sections

1. **Home** - Getting started, installation, quick start
2. **User Guide** - Comprehensive PyTorch usage documentation
3. **Examples** - Interactive notebooks and real-world use cases
4. **Developer Guide** - Contributing and development
5. **Support** - FAQ, troubleshooting, changelog

---

## 🚀 Quick Start

### View Documentation Locally

1. Install dependencies:
   ```bash
   pip install -r docs-requirements.txt
   ```

2. Serve locally:
   ```bash
   mkdocs serve
   ```

3. Open in browser:
   ```
   http://localhost:8000
   ```

### Build Documentation

```bash
mkdocs build
```

Output will be in `site/` directory.

---

## 📖 Documentation Contents

### Home Section
- **Overview** (`index.md`) - Welcome and introduction
- **Features** (`home/features.md`) - Key capabilities
- **Installation** (`home/installation.md`) - Setup guide
- **Quick Start** (`home/quickstart.md`) - Get started quickly
- **What's New** (`home/whats-new.md`) - Latest updates

### User Guide
- **Introduction** - Core concepts and workflow
- **PyTorch Backend** - Complete PyTorch documentation
  - Overview, DLBacktraceFX API, Execution engines, Operations, Tracing
- **Relevance Propagation** - Understanding explanations
  - Overview, Modes, Tasks, Parameters
- **Visualization** - Creating visualizations
- **Best Practices** - Guidelines for effective use

### Examples
- **Colab Notebooks** - 40+ interactive notebooks covering vision, NLP, and advanced models
- **Use Cases** - Real-world applications and practical implementations

### Developer Guide
- **Architecture** - System architecture overview
- **Contributing** - How to contribute
- **Layer Implementation** - Adding custom layers
- **Execution Engine** - Engine internals
- **CUDA Development** - Writing CUDA kernels
- **Testing** - Testing guidelines
- **Benchmarking** - Performance benchmarking

### Support
- **FAQ** - Frequently asked questions
- **Troubleshooting** - Common issues and solutions
- **Performance Tips** - Optimization guide
- **Known Issues** - Current limitations
- **Changelog** - Version history
- **License** - MIT License

---

## 🛠️ Building & Deploying

### Local Development

```bash
# Serve with auto-reload
mkdocs serve

# Build static site
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### Configuration

Main configuration is in `mkdocs.yml`:

```yaml
site_name: DL-Backtrace Documentation
theme:
  name: material
  palette:
    primary: indigo
nav:
  # Navigation structure
```

### Requirements

Dependencies are in `docs-requirements.txt`:
- `mkdocs>=1.5.0`
- `mkdocs-material>=9.5.0`
- `pymdown-extensions>=10.7`
- `mkdocs-minify-plugin>=0.8.0`
- `mkdocs-git-revision-date-localized-plugin>=1.2.0`

---

## ✏️ Contributing to Documentation

### Adding New Pages

1. Create markdown file in appropriate directory:
   ```bash
   touch docs/guide/new-topic.md
   ```

2. Add to navigation in `mkdocs.yml`:
   ```yaml
   nav:
     - Guide:
       - New Topic: guide/new-topic.md
   ```

### Markdown Features

#### Admonitions
```markdown
!!! tip "Tip"
    Helpful information

!!! warning "Warning"
    Important warning

!!! note "Note"
    Additional context
```

#### Code Blocks
```markdown
\`\`\`python
def example():
    return "code"
\`\`\`
```

#### Tabs
```markdown
=== "PyTorch"
    PyTorch code

=== "TensorFlow"
    TensorFlow code
```

#### Math
```markdown
Inline: \\(x^2\\)
Block: \\[E = mc^2\\]
```

---

## 📋 Documentation Checklist

### Content Completed ✅
- [x] Home section (index, features, installation, quickstart, what's new)
- [x] PyTorch user guide (overview, API, execution engines, operations, tracing)
- [x] TensorFlow user guide (overview, API, layers)
- [x] Relevance propagation guide (overview, modes, tasks, parameters)
- [x] Visualization guide
- [x] Best practices
- [x] Examples (PyTorch, TensorFlow, Colab notebooks, use cases)
- [x] Developer guide (architecture, contributing, implementation, testing)
- [x] Support (FAQ, troubleshooting, changelog, license, known issues, performance)

### Content In Progress 🔄
- [ ] API reference (stubs created, needs detailed content)
- [ ] Tutorials (stubs created, needs detailed step-by-step guides)

### Future Additions 🔮
- [ ] Video tutorials
- [ ] Interactive examples
- [ ] API auto-generation from docstrings
- [ ] More advanced use cases
- [ ] Performance comparison charts

---

## 📊 Documentation Statistics

```bash
# Count markdown files
find docs -name "*.md" | wc -l
# Result: 50+ files

# Count words
find docs -name "*.md" -exec wc -w {} + | tail -1
# Result: ~30,000+ words
```

---

## 🔗 Useful Links

- **Documentation**: http://localhost:8000 (local) or deployed URL
- **Repository**: https://github.com/aryaxai/DL-Backtrace
- **Issues**: https://github.com/aryaxai/DL-Backtrace/issues
- **MkDocs**: https://www.mkdocs.org/
- **Material Theme**: https://squidfunk.github.io/mkdocs-material/

---

## 📝 Documentation Style Guide

### Headers
- H1: Page title (once per page)
- H2: Major sections
- H3: Subsections
- H4: Minor subsections

### Code
- Use backticks for inline code: `code`
- Use code blocks for multi-line code
- Always specify language: \`\`\`python

### Links
- Relative links for internal: `[text](../other.md)`
- Absolute links for external: `[text](https://example.com)`

### Lists
- Use `-` for unordered lists
- Use `1.` for ordered lists
- Indent nested lists with 4 spaces

---

## 🤝 Getting Help

- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: support@aryaxai.com

---

## 📄 License

Documentation is part of DL-Backtrace and is licensed under the MIT License.

---

<div align="center">

**Ready to explore the documentation?**

```bash
mkdocs serve
```

Then visit http://localhost:8000

</div>



