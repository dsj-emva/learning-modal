"""Context layer (Phase 6): an LLM reads each lead card against a business brief and returns named enum judgments.

``card`` (what the agent may see), ``contract`` (what it must return), ``agent`` (SDK runtime), ``cache`` (replies
keyed by card, brief, prompt version, prompt fingerprint
and model), ``features`` (judgments to fixed-schema design columns). ADR 0014.
"""
