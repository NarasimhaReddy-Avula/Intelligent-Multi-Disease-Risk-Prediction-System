"""LLM utilities.

This package contains lightweight, provider-agnostic clients for calling online LLM APIs
(without downloading local model weights).

Design goals:
- Keep secrets out of the repo: read API keys from environment variables.
- Avoid heavy dependencies: use `requests`.
- Prefer structured outputs (JSON) to reduce hallucination.
"""
