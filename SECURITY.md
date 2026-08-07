# Security policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not open a public issue containing credentials, private endpoints, or exploitable details.

## Secrets

The project reads provider credentials only from environment variables. Local `.env*`, `.deepseek_key`, and `.gemini_key` files are ignored and must never be committed. If a credential is accidentally published, revoke it before removing it from Git history.

## Public demo boundary

The Sites demos do not accept or store visitor model credentials and do not expose a live generation endpoint.
