"""Auto-discovery loader for skills package."""

from biobank_agent.registry import autodiscover_skills

# Skills are loaded via autodiscover_skills() in agent.py
# Each skill file uses the @skill decorator to self-register.
