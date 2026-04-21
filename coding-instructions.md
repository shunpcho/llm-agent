# Style Guide

## Table of Contents

- [Introduction](#introduction)
- [Style](#style)
  - [Descriptive variable names](#descriptive-variable-names)
  - [Code formatting](#code-formatting)
  - [Code linting](#code-linting)
  - [Code typing](#code-typing)
  - [Python idioms](#python-idioms)
- [Logic](#logic)
- [Comments & docstrings](#comments--docstrings)
- [Testing](#testing)
  - [Writing tests](#writing-tests)
  - [Test types](#test-types)

## Introduction

This document describes coding conventions for Python code.

For anything not covered in this document, refer to the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## Style

### Descriptive variable names

Naming things is hard. The rules below aim to reduce guesswork:

- Do not use one-character variable names.
  - Exceptions: `x`, `y`, and `z` as coordinates.
  - Single-character names are hard to search for and provide no context about the variable’s purpose.
- Do not shadow built-in functions.
  - Exception: `id`. It is commonly used, and alternatives can feel artificial.
- Acronyms, abbreviations, or other short forms should be explained using
  [tagref](https://github.com/stepchowfun/tagref), unless they are almost universally understood.

### Code formatting

All code submitted should be formatted using `ruff format`.

### Code linting

All code submitted to the project should be checked with:

- [`ruff check`](https://docs.astral.sh/ruff/linter/)
- [`pyright`](https://github.com/microsoft/pyright)

### Code typing

This project uses **Pyright in strict mode**, with a few repository-specific relaxations configured in `pyproject.toml`.
Write code that is type-safe and works with these settings.

#### General rules

- Add type hints for public functions/methods/classes.
- Prefer explicit, simple types. Avoid `Any` unless unavoidable.
- Prefer `TypeAlias`, `Literal`, `Final`, `Protocol`, and other typing utilities where they improve clarity.
- Always use `pathlib.Path` for paths.
- For NumPy arrays, prefer `npt.NDArray[...]` to document dtype expectations.
  - Example: distinguish raw images (`np.uint8`) vs normalized tensors/arrays (`np.float32`).

#### Suppressing type errors (preferred approach)

Avoid suppressions when possible. If you must suppress:

- Prefer **narrow, documented suppressions** over blanket ignores.
- Prefer Pyright-specific ignores with a diagnostic code:
  - ✅ `# pyright: ignore[reportUnknownVariableType]  # <reason>`
  - ✅ `# pyright: ignore[reportGeneralTypeIssues]  # <reason>`
- Avoid blanket ignores:
  - ❌ `# type: ignore`
  - ❌ `# pyright: ignore`

If you use `cast(...)`, `Any`, or an ignore comment, add a short justification explaining why the type cannot be expressed more precisely.

> Note: The repository currently disables some diagnostics (e.g. unknown member types / missing stubs) in `pyproject.toml`.
> Do not “fight” these settings in code; rely on them where appropriate, but still prefer precise types where feasible.

#### Interaction with library stubs

Because `reportMissingTypeStubs` may be disabled in this repo, third-party libraries may appear less typed.
Even so:

- Type your own API surfaces precisely.
- Consider adding lightweight wrapper types or protocols around poorly typed libraries rather than spreading `Any`.

### Python idioms

- When using a `dataclass`, set `slots=True`.
  - This reduces memory usage and prevents accidental creation of new attributes.
- When using `zip`, use `strict=True`.
  - If setting it to `False`, add a comment explaining why.
  - This catches length mismatches that could otherwise cause silent bugs.
- Prefer `a / b` over `Path(a, b)` for concatenating paths.
  - It is more readable and consistent with `pathlib`’s design.

## Logic

- Validate function/program inputs at boundaries.
  - Fail fast with clear error messages.
- Avoid side effects (e.g., mutating arguments). Prefer pure functions where practical.

## Comments & docstrings

- Do not use comments (`# ...`) instead of docstrings (`"""..."""`).
  - This applies to module-level “comments” and dataclass attribute documentation.

```python
# Bad
# This is what the module does


@dataclass
class Foo:
    bar: int  # The bar value


# Good
"""This is what the module does."""


@dataclass
class Foo:
    bar: int
    """The bar value"""
```

- Use docstrings and comments where they bring value; do not feel obligated to document everything.
  - Documenting obvious things (e.g., `input: the input`) creates bloat and makes code harder to read.
- Complex algorithms should have explanatory comments.
- Do not put type hints in docstrings:
  - They are not used by the IDE/type checker.
  - They duplicate information from the function signature.
  - They can get out of sync with the function signature and cannot be checked automatically.
  - This applies to both the `Args` and `Returns` sections.

## Testing

The purpose of testing is to prevent regressions. When changes are made to source code, automated testing ensures there are no unexpected impacts on other parts of the codebase, which improves development speed.

Before merging a PR, tests should be executed by CI and must pass.

### Writing tests

- Write tests using [pytest](https://docs.pytest.org/en/stable/). Do not use the standard library’s testing modules.
- Do not aim for a test coverage percentage. Focus on writing useful tests.
  - Simple functions that are deemed unlikely to break do not need tests.
- When fixing a bug, add a non-regression test.
- Do not use assertion messages in tests. See the pytest docs:
  <https://docs.pytest.org/en/7.1.x/how-to/assert.html#asserting-with-the-assert-statement>.

### Test types

- **Unit tests**
  - Tests that can be performed with only the target module.
- **Integration tests**
  - Tests that require external modules or systems.
- **System tests**
  - Tests that verify customer usage validity, including UI and performance.
  - Create separate test types for different purposes, such as model accuracy verification tests using customer data.