from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'star_classification.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

# ID columns are loaded as strings: they label things, they are not measurements
ID_COLS = ['obj_ID', 'spec_obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'plate', 'fiber_ID']

FILTERS = ['u', 'g', 'r', 'i', 'z']
COLORS = ['u_g', 'g_r', 'r_i', 'i_z']

# Final feature set: 4 colors + r (same information as all five filters, chosen by cross-validation)
FINAL_FEATURES = COLORS + ['r']

TARGET = 'class'
CLASSES = ['GALAXY', 'STAR', 'QSO']

# -9999 is a placeholder for no measurement
PLACEHOLDER = -9999

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
