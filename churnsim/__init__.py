"""churnsim: discrete-event simulation of pipeline-parallel inference under node churn.

Reproduces the fault-tolerance experiment of Petals (Borzunov et al., arXiv:2312.08361,
Table 1 / Appendix F): sequential token generation through a pipeline of stages where
each hop fails with probability p, compared across three recovery strategies.
"""

from .model import Config, Strategy, simulate, run_many  # noqa: F401
