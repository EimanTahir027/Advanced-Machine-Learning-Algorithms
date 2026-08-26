# Security Policy

## Supported Versions

This repository is a collection of educational notebooks and does not follow a formal release/versioning cycle. The `main` branch is the only actively maintained version, and any security-related fixes will be applied there.

| Branch | Supported          |
| ------ | ------------------ |
| main   | :white_check_mark: |
| other  | :x:                |

## Reporting a Vulnerability

If you discover a security issue in this repository — for example, a dependency with a known vulnerability, unsafe code execution in a notebook, or exposed credentials — please report it responsibly:

- **Preferred:** Open a [private security advisory](../../security/advisories/new) on GitHub (Security tab → "Report a vulnerability").
- **Alternative:** Open a regular issue with minimal detail and a note that it concerns a security matter, and a maintainer will follow up to arrange a private channel.

Please avoid publicly disclosing details of the vulnerability until it has been reviewed.

**What to expect:**
- An initial response within **5–7 days** acknowledging your report.
- An assessment of whether the issue is valid and its potential impact.
- A fix or mitigation, where applicable, along with credit to the reporter (unless anonymity is requested).
- If the report is declined (e.g. not a genuine vulnerability, or out of scope for a notebook-based educational repo), an explanation will be provided.

## Scope

As this repository primarily contains Jupyter notebooks for learning purposes, typical concerns include:
- Vulnerable or outdated third-party packages (e.g. via `requirements.txt`)
- Unsafe deserialization or execution of untrusted data within notebooks
- Accidental exposure of API keys, tokens, or credentials in committed files

General bugs or incorrect model implementations are not security issues — please open a standard issue for those instead.
