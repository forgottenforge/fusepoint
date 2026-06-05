# Security Policy

## Supported versions

| Version  | Supported               |
|----------|-------------------------|
| 1.0.x    | active                  |
| < 1.0    | no longer supported     |

## Reporting a vulnerability

If you discover a security vulnerability in FUSE (`fusepoint`), please
**report it privately** so that it can be fixed before public disclosure.

**Do not open a public GitHub issue for security problems.**

### How to report

Email: **[nfo@forgottenforge.xyz](mailto:nfo@forgottenforge.xyz)**
Subject prefix: `[security] fusepoint: <short summary>`

Please include:

- A clear description of the vulnerability and its impact.
- Steps to reproduce or a minimal proof-of-concept.
- The FUSE version and environment (OS, Python version).
- Your name / handle for credit, or "anonymous" if you prefer.

### What to expect

| Step | Timeline |
|------|----------|
| Acknowledgement of receipt | within 5 business days |
| Initial triage and severity assessment | within 10 business days |
| Patch development | depends on severity; tracked privately |
| Coordinated disclosure and release | once a fix is available |

We will work with you on a coordinated disclosure timeline. We do not
operate a bug-bounty programme, but we credit responsible reporters in
release notes unless you ask us not to.

## Scope

In scope:

- Code-execution, deserialization, or path-traversal vulnerabilities in
  the `fusepoint/` package or the bundled Streamlit web UI.
- Unsafe defaults or documentation that would lead a downstream user
  into a credential-leak situation.
- Issues in the bundled web demo (`fuse-ui` / Streamlit app) that allow
  data exfiltration across user sessions.
- Dependency-supply-chain issues that we should pin or pull-request
  upstream.

Out of scope:

- Bugs in third-party libraries we depend on (please report those
  upstream — including `sigma-c-framework`, `streamlit`, `pandas`,
  `scipy`).
- Denial of service caused by user-supplied pathological inputs that
  are well-documented as out-of-scope (e.g. unbounded `analyze()` calls
  on infinite arrays).
- Statistical interpretation disagreements about FUSE's stability score;
  those belong in regular issues or academic correspondence.

## Disclosure history

No security issues have been reported to date.
