# Contributing to [Your Project Name]

First off, thank you for considering contributing to QPIXL! We appreciate any help, whether it's reporting a bug, proposing a new feature, or writing code.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/moth-quantum/QuantumArtHack.git
    cd QuantumArtHack
    ```
3.  **Create a virtual environment** and activate it. This keeps dependencies isolated.
    ```bash
    python -m venv .venv
    # On Windows:
    # .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```
4.  **Install development dependencies**:
    ```bash
    pip install -e .
    pip install black ruff
 **Python Environments with** `pyproject.toml`

The `pyproject.toml` file is a modern standard for configuring Python projects, including managing dependencies and build systems. Several tools can work with it to manage your project's virtual environment.

Poetry is a popular tool for dependency management and packaging in Python. It uses `pyproject.toml` to manage project metadata, dependencies, and create isolated environments.

###  Installing Poetry

If you don't have Poetry installed, you can install it using the official installer:

**macOS / Linux / WSL:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Using poetry 
in the directory of the project:
```poetry install```

### Alternatively
just look at the list of packages in `pyproject.toml` and install them manually - there aren't many dependencies. 


5.  **Create a new branch** for your changes:
    ```bash
    git checkout -b feature/your-awesome-feature
    # or
    git checkout -b fix/bug-description
    ```

## Making Changes

1.  Make your code changes.
2.  **Format your code** using Black:
    ```bash
    black .
    ```
3.  **Lint your code** using Ruff to catch common issues and ensure style consistency:
    ```bash
    ruff check .
    # You can also ask ruff to try and fix issues automatically:
    # ruff check . --fix
    ```
4.  Add or update tests if you're adding new features or fixing bugs. (Even if your project doesn't have extensive tests yet, consider how you might test your change).
5.  Ensure your changes don't break existing functionality.

## Committing Your Changes

*   Write clear, concise commit messages. A good format is:
    *   A short summary 
    *   A blank line
    *   More detailed explanatory text, if necessary.
    *   Example: `Fix: Resolve issue with parsing input dates`

## Submitting a Pull Request

1.  Push your branch to your fork on GitHub:
    ```bash
    git push origin feature/your-awesome-feature
    ```
2.  Open a Pull Request (PR) from your fork to the main `QPIXL` repository.
3.  Provide a clear title and description for your PR:
    *   Explain the problem you're solving or the feature you're adding.
    *   Reference any relevant issues (e.g., "Closes #123").
4.  Be prepared to discuss your changes and make adjustments if requested by the maintainers.

## Code Style

We use **Black** for auto-formatting Python code and **Ruff** for linting. Please run these tools before committing your changes. This helps maintain a consistent and readable codebase.

## Reporting Bugs or Requesting Features

*   Use the GitHub Issues section of the project.
*   For bug reports, please include:
    *   Steps to reproduce the bug.
    *   Expected behavior.
    *   Actual behavior.
    *   Your Python version and operating system.
*   For feature requests, please describe the feature and why you think it would be a valuable addition.

## Questions?

If you have any questions, feel free to open an issue and ask!

---

Thank you for contributing!