## CONTRIBUTING

### 1. Find an Issue

- Browse [open issues](../../issues)
- Check the `ISSUES.md` file for suggested tasks with bounty points
- Look for `good-first-issue` label if you're new

### 2. Claim an Issue

1. Comment on the issue: **"I would like to work on this"**
2. Wait for maintainer approval (we'll assign you)
3. Start working only after assignment
4. If no response within 24 hours, ping maintainers

### 3. Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/pokedex.git
cd pokedex
```

### 4. Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv poke
source sudoku/bin/activate

# Install dependencies (optional, if it's available)
pip install -r requirements.txt

```

### 5. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

**Branch naming conventions:**
- `feature/` - for new features
- `fix/` - for bug fixes
- `refactor/` - for code refactoring
- `docs/` - for documentation updates
- `test/` - for adding tests

### 6. Testing

- Write tests for all new features
- Ensure existing tests pass before submitting PR
- Run tests: `pytest` or `python -m pytest`
- Aim for >80% code coverage

### 7. Commits

**Write clear commit messages:**

**Commit message prefixes:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding/updating tests
- `refactor:` - Code refactoring
- `style:` - Formatting, missing semicolons, etc.
- `chore:` - Maintenance tasks

## Submitting a Pull Request

### Before Submitting

-  Code follows style guidelines
-  All tests pass locally
- New features have tests
- Documentation is updated
- Commit messages are clear
- No unnecessary files included (.pyc, __pycache__, etc.)

### Create Pull Request

1. Push your branch:
```bash
git push origin feature/your-feature-name
```

2. Go to GitHub and click "New Pull Request"

3. Fill in the PR template:

## Description
Brief description of changes

## Related Issue
Fixes #issue_number

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Tests pass locally
- [ ] New tests added (if applicable)

## Screenshots (if applicable)
Add screenshots or GIFs for UI changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed my code
- [ ] Commented complex code sections
- [ ] Updated documentation
- [ ] No breaking changes
```

4. Wait for review and address feedback
