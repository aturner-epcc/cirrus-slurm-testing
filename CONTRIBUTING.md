# Contributing to Cirrus Slurm Test Harness

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/cirrus-slurm-harness.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `source venv/bin/activate`
5. Install development dependencies: `pip install -e ".[dev]"`

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `tests/description` - Test additions or improvements

### Commit Messages

Follow conventional commits format:

```
feat: add support for new partition type
fix: correct charge calculation for hetjobs
docs: update README with examples
tests: add integration tests for job arrays
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use black for formatting: `black src/ tests/`
- Use ruff for linting: `ruff check src/ tests/`

### Testing

Run tests before submitting a PR:

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=cirrus_harness --cov-report=html

# Specific test suite
pytest tests/test_job_types.py -v
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new classes and methods
- Include examples in docstrings for complex functionality
- Update API reference sections as needed

## Pull Request Process

1. **Create a branch** from `main`
2. **Make your changes** with clear commits
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Run all tests** and ensure they pass
6. **Submit a pull request** with:
   - Clear description of changes
   - Reference to any related issues
   - Examples of usage (if applicable)

## Code Review

All PRs will be reviewed for:

- Correctness and functionality
- Code style and consistency
- Test coverage
- Documentation quality
- Performance implications

## Reporting Issues

### Bug Reports

Include:
- Python version
- Cirrus system information
- Steps to reproduce
- Expected vs actual behavior
- Error messages and tracebacks
- Example code (if applicable)

### Feature Requests

Include:
- Use case description
- Expected behavior
- Examples of usage
- Priority/importance

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open an Issue
- **Cirrus-specific questions**: Contact support@cirrus.ac.uk

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
