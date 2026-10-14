"""Limits and defaults shared across the core. Kept in one place so the
README, tests and API can reference the same numbers."""

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_ROWS = 200_000
MAX_COLUMNS = 300
MIN_ROWS = 10

ALLOWED_EXTENSIONS = (".csv", ".tsv", ".txt")

# Extra strings treated as missing on top of pandas' defaults ("", "NA", "null", ...).
EXTRA_NA_VALUES = ["?", "-", "--"]  # deliberately NOT "none": it is often a real category

# Target rules
MAX_CLASSES = 50
MIN_ROWS_PER_CLASS = 2  # needed so a stratified split can put each class on both sides
SMALL_CLASS_WARNING = 10
IMBALANCE_WARNING_RATIO = 0.35  # minority/majority below this -> warn (binary: minority < ~26%)

# Feature typing
CATEGORICAL_MAX_UNIQUE_RATIO = 0.5
TEXT_MIN_AVG_LENGTH = 40
LOW_CARDINALITY_NUMERIC = 15  # integer columns with <= this many values may be categories
ONE_HOT_MAX_CATEGORIES = 30
LEAKAGE_ASSOCIATION_THRESHOLD = 0.9

# Training
DEFAULT_TEST_SIZE = 0.2
MIN_TEST_SIZE = 0.1
MAX_TEST_SIZE = 0.5
DEFAULT_RANDOM_STATE = 42
CV_MAX_TRAIN_ROWS = 50_000
PERMUTATION_MAX_ROWS = 2_000
PERMUTATION_REPEATS = 5

# Store
MAX_DATASETS = 20
MAX_MODELS = 60
