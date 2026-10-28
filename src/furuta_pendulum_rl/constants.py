from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PENDULUM_MODEL_PATH = str(PROJECT_ROOT / "furuta_model" / "furuta.xml")
RL_MODEL_PATH = str(PROJECT_ROOT / "rl_model" / "furuta_rl_model")
VEC_NORMALIZE_PATH = str(PROJECT_ROOT / "rl_model" / "vec_normalize.pkl")