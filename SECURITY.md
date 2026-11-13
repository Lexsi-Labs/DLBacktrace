# Security Policy

## Supported Versions

Security fixes are released for the latest published version of DL-Backtrace only. Please upgrade to the latest release before reporting an issue.

| Version        | Supported          |
| -------------- | ------------------ |
| Latest release | :white_check_mark: |
| Older releases | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report privately using GitHub's private vulnerability reporting: [open a report](https://github.com/Lexsi-Labs/DLBacktrace/security/advisories/new), or open the repository's **Security** tab and choose **Report a vulnerability**.

### What to Include

1. A description of the vulnerability
2. Steps to reproduce the issue
3. The potential impact
4. Affected versions (if known)
5. A suggested fix (if you have one)

### What to Expect

- Acknowledgment of your report within 7 days
- An assessment of severity and an update on our progress
- A fix released as a patch version, noted in the [changelog](docs/support/changelog.md)
- Credit for the report in the advisory, with your permission

## Disclosure Policy

We will prepare a security advisory, release a patch, and disclose the issue publicly after users have had time to update.

## Security Practices

- Only load models, checkpoints and datasets from sources you trust. Loading a checkpoint can run code.
- `dlbacktrace-server` has no authentication. It binds to `127.0.0.1` by default; use `--host 0.0.0.0` only on a trusted network, and keep request cache directories under `--cache-root`.
- Never commit API keys or tokens, such as a Hugging Face token.
- Dependencies and GitHub Actions are monitored with Dependabot, and the code is scanned with CodeQL.
