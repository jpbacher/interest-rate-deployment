from pathlib import Path

import lending_model


PACKAGE_ROOT = Path(lending_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT/'trained_models'
DATASET_DIR = PACKAGE_ROOT/'datasets'

# data
TRAINING_DATA_FILE = "lending_data.csv"
TARGET = 'int_rate'

# input features
FEATURES = [
    'loan_amnt',
    'funded_amnt_inv',
    'installment',
    'dti',
    'mths_since_last_delinq',
    'revol_bal',
    'revol_util',
    'tot_coll_amt',
    'mths_since_rcnt_il',
    'il_util',
    'max_bal_bc',
    'all_util',
    'total_rev_hi_lim',
    'acc_open_past_24mths',
    'avg_cur_bal',
    'bc_open_to_buy',
    'delinq_amnt',
    'mo_sin_old_il_acct',
    'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op',
    'mo_sin_rcnt_tl',
    'mths_since_recent_bc',
    'mths_since_recent_inq',
    'pct_tl_nvr_dlq',
    'percent_bc_gt_75',
    'total_il_high_credit_limit'
]

# variables to remove '%' in string & convert to numeric data type
VARS_WITH_PERC_SIGNS = ['revol_util']

# numerical variables with NA in training set
NUM_VARS_WITH_NA = [
    'dti',
    'revol_util',
    'mths_since_rcnt_il',
    'all_util',
    'avg_cur_bal',
    'bc_open_to_buy',
    'mo_sin_old_il_acct',
    'mths_since_recent_bc',
    'mths_since_recent_inq',
    'mths_since_last_delinq',
    'il_util',
    'mths_since_recent_inq',
    'percent_bc_gt_75'
]

NUMERICAL_NA_NOT_ALLOWED = [
    feature for feature in FEATURES if feature not in NUM_VARS_WITH_NA
]

PIPELINE_NAME = 'rf_lending_model'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'

ACCEPTABLE_MODEL_DIFFERENCE = 0.05