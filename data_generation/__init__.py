"""
Generate .DATA and Schedule files for Eclipse simulations based off AGSI data.
"""
from .eclipse_ng_gen import generate_all_scripts  # noqa: F401, F403
from .synthetic_script import generate_full_script  # noqa: F401, F403
