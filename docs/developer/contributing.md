# Contributing to DL-Backtrace

We welcome contributions from the community! This guide will help you get started.

---

## Ways to Contribute

### 🐛 Report Bugs
Found a bug? [Create an issue](https://github.com/Lexsi-Labs/DLBacktrace/issues/new)

### 💡 Suggest Features
Have an idea? [Open a feature request](https://github.com/Lexsi-Labs/DLBacktrace/issues/new)

### 📝 Improve Documentation
Documentation improvements are always welcome!

### 💻 Submit Code
Fix bugs or implement features via pull requests.

### 🧪 Add Tests
Help improve test coverage.

### 📚 Create Examples
Share your use cases and examples.

---

## Getting Started

### 1. Fork the Repository

```bash
# Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/DL-Backtrace.git
cd DL-Backtrace
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---

## Development Workflow

### 1. Make Changes

Edit the relevant files in `dl_backtrace/`.

### 2. Test Your Changes

```bash
# Run existing tests
python -m pytest tests/

# Test with example
python benchmarks/trace_RoBERTa.py
```

### 3. Follow Code Style

We follow Python PEP 8 style guidelines:

```python
# Good
def calculate_relevance(input_tensor, weights):
    """Calculate relevance scores."""
    return input_tensor @ weights

# Use descriptive names
# Add docstrings
# Keep functions focused
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: Add support for new operation"
# or
git commit -m "fix: Fix memory leak in execution engine"
```

**Commit Message Format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `perf:` Performance improvements

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## Contributing Areas

### Adding New Operations

To add support for a new PyTorch operation:

1. **Locate the execution engine:**
   ```
   dl_backtrace/pytorch_backtrace/dlbacktrace/core/execution_engine_noncache.py
   ```

2. **Add operation handler:**
   ```python
   elif func_name == "your_new_op":
       # Get inputs
       layer_in = [...]
       
       # Process
       output = aten_op(layer_in, **hyperparams)
       
       return output
   ```

3. **Test the operation:**
   ```python
   # Create test model
   class TestModel(nn.Module):
       def forward(self, x):
           return your_new_op(x)
   
   # Test tracing
   model = TestModel()
   dlb = DLBacktrace(model, input_for_graph=(dummy,))
   ```

4. **Add to documentation:**
   Update `docs/guide/pytorch/operations.md`

### Adding Layer Implementations

For custom layer relevance rules:

1. **Create layer directory:**
   ```
   dl_backtrace/pytorch_backtrace/dlbacktrace/utils/cuda_utils/YourLayer/
   ```

2. **Implement versions:**
   - `original_version.py` - NumPy baseline
   - `pytorch_version.py` - PyTorch implementation
   - `cuda_version/` - CUDA implementation (optional)

3. **Add launcher:**
   Update `utils/default_v2.py`

### Improving Documentation

Documentation is in `docs/` using MkDocs:

```bash
# Install MkDocs
pip install mkdocs mkdocs-material

# Serve locally
cd DL-Backtrace
mkdocs serve

# View at http://localhost:8000
```

Edit Markdown files in `docs/` and see changes live.

### Adding Examples

Add example notebooks:

1. Create Colab notebook
2. Test thoroughly
3. Add link to `docs/examples/colab-notebooks.md`
4. Update README.md

---

## Pull Request Guidelines

### Before Submitting

✅ Code follows project style  
✅ Tests pass  
✅ Documentation updated  
✅ Commit messages are clear  
✅ Branch is up to date with main

### PR Description

Include:

- **What**: What does this PR do?
- **Why**: Why is this change needed?
- **How**: How does it work?
- **Testing**: How was it tested?
- **Breaking Changes**: Any breaking changes?

### Example PR Description

```markdown
## Add support for GroupNorm operation

### What
Adds support for `torch.nn.GroupNorm` in the execution engine.

### Why
Users reported that models with GroupNorm fail to trace.

### How
- Added `group_norm` handler in execution engine
- Follows same pattern as `layer_norm`
- Handles all hyperparameters correctly

### Testing
- Tested with ResNet models using GroupNorm
- Added unit test in `tests/test_operations.py`
- Verified with example in `examples/groupnorm_example.py`

### Breaking Changes
None. Fully backward compatible.
```

---

## Code Review Process

### Review Timeline

- Initial review: Within 1 week
- Follow-up reviews: Within 2-3 days
- Merge: After approval from maintainers

### What We Look For

- **Correctness**: Does it work as intended?
- **Quality**: Is the code clean and maintainable?
- **Testing**: Is it adequately tested?
- **Documentation**: Is it documented?
- **Compatibility**: No breaking changes without discussion

### Addressing Feedback

```bash
# Make requested changes
git add .
git commit -m "Address review feedback"
git push origin feature/your-feature-name
```

---

## Testing Guidelines

### Writing Tests

```python
# tests/test_your_feature.py
import torch
import pytest
from dl_backtrace.pytorch_backtrace import DLBacktrace

def test_your_feature():
    """Test description."""
    # Arrange
    model = create_test_model()
    input_tensor = torch.randn(1, 10)
    
    # Act
    dlb = DLBacktrace(model, input_for_graph=(input_tensor,))
    result = dlb.predict(input_tensor)
    
    # Assert
    assert result is not None
    assert len(result) > 0
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_operations.py

# Specific test
pytest tests/test_operations.py::test_linear_operation

# With coverage
pytest --cov=dl_backtrace tests/
```

---

## Documentation Style

### Docstrings

Use Google-style docstrings:

```python
def calculate_relevance(input_tensor, weights, bias=None):
    """Calculate relevance scores for a linear layer.
    
    Args:
        input_tensor: Input tensor of shape (batch, features)
        weights: Weight matrix of shape (features, out_features)
        bias: Optional bias vector of shape (out_features,)
    
    Returns:
        Relevance tensor of shape (batch, features)
    
    Raises:
        ValueError: If tensor shapes are incompatible
    
    Example:
        >>> input_tensor = torch.randn(32, 128)
        >>> weights = torch.randn(128, 64)
        >>> relevance = calculate_relevance(input_tensor, weights)
        >>> print(relevance.shape)
        torch.Size([32, 128])
    """
    pass
```

### Comments

```python
# Good comments explain WHY
# Calculate weighted contribution (not just "multiply weights")
contribution = weights * activations

# Use comments for complex logic
# Use docstrings for functions/classes
```

---

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features (backward compatible)
- **Patch** (0.0.X): Bug fixes

### Creating a Release

(For maintainers only)

1. Update version in `setup.py` and `version.py`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag -a v2.0.0 -m "Release 2.0.0"`
4. Push tag: `git push origin v2.0.0`
5. Create GitHub release

---

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Assume good intentions
- Provide constructive feedback
- Focus on the issue, not the person

### Getting Help

- **Questions**: [GitHub Discussions](https://github.com/Lexsi-Labs/DLBacktrace/discussions)
- **Issues**: [GitHub Issues](https://github.com/Lexsi-Labs/DLBacktrace/issues)
- **Email**: [support@lexsi.ai](mailto:support@lexsi.ai)

---

## Recognition

Contributors are recognized in:

- README.md contributors section
- Release notes
- Changelog

Thank you for contributing to DL-Backtrace! 🎉

---

## Quick Reference

```bash
# Setup
git clone https://github.com/YOUR_USERNAME/DL-Backtrace.git
cd DL-Backtrace
pip install -e .

# Create branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "feat: My new feature"

# Push and create PR
git push origin feature/my-feature
# Then create PR on GitHub

# Keep branch updated
git checkout main
git pull upstream main
git checkout feature/my-feature
git rebase main
```

---

## Additional Resources

- [Developer Guide](architecture.md) - Architecture overview
- [Testing Guide](testing.md) - Testing guidelines
- [Code Style Guide](https://pep8.org/) - PEP 8

---

<div align="center">

**Ready to contribute?**

[Create an Issue →](https://github.com/Lexsi-Labs/DLBacktrace/issues/new){ .md-button }
[Fork Repository →](https://github.com/Lexsi-Labs/DLBacktrace/fork){ .md-button }

</div>



