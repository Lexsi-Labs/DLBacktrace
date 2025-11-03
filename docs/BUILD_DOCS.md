# Building Documentation

Guide for building and deploying DL-Backtrace documentation.

---

## Prerequisites

Install MkDocs and dependencies:

```bash
pip install -r docs-requirements.txt
```

---

## Local Development

### Serve Locally

```bash
cd DL-Backtrace
mkdocs serve
```

Then open http://localhost:8000 in your browser.

Changes to markdown files are automatically reflected.

### Build Documentation

```bash
mkdocs build
```

This creates a `site/` directory with the static HTML.

---

## Project Structure

```
DL-Backtrace/
├── mkdocs.yml              # Configuration
├── docs/                   # Documentation source
│   ├── index.md           # Home page
│   ├── home/              # Getting started
│   ├── guide/             # User guides
│   ├── tutorials/         # Step-by-step tutorials
│   ├── examples/          # Examples and notebooks
│   ├── api/               # API reference
│   ├── developer/         # Developer documentation
│   ├── support/           # FAQ, troubleshooting
│   └── javascripts/       # Custom JS (MathJax)
└── site/                  # Built documentation (generated)
```

---

## Adding Content

### Create New Page

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
!!! tip "Helpful Tip"
    This is a tip!

!!! warning "Warning"
    Be careful!

!!! note "Note"
    Important information.
```

#### Code Blocks

```markdown
\`\`\`python
def example():
    return "Hello"
\`\`\`
```

#### Tabs

```markdown
=== "PyTorch"
    PyTorch code here

=== "TensorFlow"
    TensorFlow code here
```

#### Math

```markdown
Inline: \(x^2\)
Block: \[E = mc^2\]
```

---

## Deployment

### GitHub Pages

```bash
# Deploy to gh-pages branch
mkdocs gh-deploy
```

### Custom Domain

Add `CNAME` file to `docs/`:

```bash
echo "docs.example.com" > docs/CNAME
```

---

## Style Guide

### Headers

```markdown
# Page Title (H1 - once per page)

## Section (H2)

### Subsection (H3)

#### Sub-subsection (H4)
```

### Links

```markdown
[Link text](relative/path.md)
[External link](https://example.com)
```

### Images

```markdown
![Alt text](images/screenshot.png)
```

### Code References

```markdown
Use backticks for `code` and functions like `DLBacktrace()`.
```

---

## Checking Links

```bash
# Check for broken links (if installed)
linkchecker http://localhost:8000
```

---

## Updating Theme

Theme is configured in `mkdocs.yml`:

```yaml
theme:
  name: material
  palette:
    primary: indigo
    accent: indigo
```

See [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) for customization options.

---

## Troubleshooting

### Build Errors

```bash
# Verbose output
mkdocs build --verbose

# Clean build
rm -rf site/
mkdocs build
```

### Port Already in Use

```bash
# Use different port
mkdocs serve -a localhost:8001
```

---

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)

---

## Maintenance

### Keeping Dependencies Updated

```bash
pip install --upgrade -r docs-requirements.txt
```

### Checking for Broken Links

Periodically check internal links are still valid.

### Updating Content

Review and update documentation when:
- New features are added
- APIs change
- User feedback suggests improvements
- Errors are found

---

<div align="center">

**Ready to contribute to documentation?**

[Edit on GitHub →](https://github.com/Lexsi-Labs/DLBacktrace/tree/main/docs){ .md-button }

</div>



