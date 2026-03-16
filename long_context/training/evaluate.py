"""
long_context/training/evaluate.py

Thin redirect to the shared evaluate.py in the project root's training/ package.

Why this file exists:
    long_context/training/ shadows ShifaMind_Local/training/ in sys.path (so that
    the experimental loss.py takes priority).  But Python won't look further down
    sys.path for a submodule that belongs to an already-resolved package, so
    `from training.evaluate import ...` would fail without this file.

    We load the root evaluate.py via importlib using its absolute path, bypassing
    the package-name collision entirely.  When that module executes
    `from training.loss import MultiObjectiveLoss`, sys.path has long_context/
    first, so it correctly picks up the experimental loss — which is what we want.
"""
import importlib.util
from pathlib import Path

_root_eval_path = Path(__file__).resolve().parent.parent.parent / "training" / "evaluate.py"
_spec = importlib.util.spec_from_file_location("_root_training_evaluate", str(_root_eval_path))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

evaluate_phase1 = _mod.evaluate_phase1
evaluate_phase2 = _mod.evaluate_phase2
evaluate_phase3 = _mod.evaluate_phase3
