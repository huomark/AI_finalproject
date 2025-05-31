# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: ai-hw
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Preliminary

# %%
import os
import pandas as pd
import re

from loguru import logger
import sys

from pygments.lexers import CppLexer
from pygments.token import (
    Token,
    Comment,
    Text,
    Keyword,
    Name,
    Literal,
    Punctuation,
    Operator,
)
import ast

from sklearn.preprocessing import MultiLabelBinarizer
import joblib

from transformers import AutoTokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# ### Configuration

# %%
PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
SUBMISSIONS_DIR = os.path.join(DATA_DIR, "submissions")
TAG_DIR = os.path.join(DATA_DIR, "tags")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")

if not os.path.exists(SUBMISSIONS_DIR):
    logger.error(f"Submissions directory not found at {SUBMISSIONS_DIR}")
if not os.path.exists(TAG_DIR):
    logger.error(f"Tags directory not found at {TAG_DIR}")


# %% [markdown]
# ### Load Code and Tags

# %%
data_records = []

for submission_filename in os.listdir(SUBMISSIONS_DIR):
    if not submission_filename.endswith(".txt"):
        continue

    problem_id = submission_filename.replace(".txt", "")
    tag_filename = submission_filename
    code_filepath = os.path.join(SUBMISSIONS_DIR, submission_filename)
    tag_filepath = os.path.join(TAG_DIR, tag_filename)

    code_content = None
    tag_list = []

    # Read source code
    try:
        with open(code_filepath, "r", encoding="utf-8") as code_file:
            code_content = code_file.read()
    except FileNotFoundError:
        logger.error(f"Code file {code_filepath} not found, but was listed")
        continue
    except Exception as e:
        logger.error(f"Could not read code file {code_filepath}: {e}")
        continue

    # Read tags
    try:
        with open(tag_filepath, "r", encoding="utf-8") as tag_file:
            tag_content = tag_file.read()
            tag_list = re.split(
                r"\s*,\s*", tag_content.strip()
            )  # Split by commas and strip whitespace
    except FileNotFoundError:
        logger.error(f"Tag file {tag_filepath} not found, but was listed")
        continue
    except Exception as e:
        logger.error(f"Could not read tag file {tag_filepath}: {e}")
        continue

    if code_content is not None:
        data_records.append(
            {
                "problem_id": problem_id,
                "raw_code": code_content,
                "tags": tag_list,
            }
        )

df = pd.DataFrame(data_records)

if not df.empty:
    logger.info(f"Successfully loaded {len(df)} records.")
    logger.info("Sample of loaded data:")
    print(df.head())
else:
    logger.warning(
        "No data records were loaded. Please check your `SUBMISSIONS_DIR` and `TAGS_DIR` paths and content."
    )


# %% [markdown]
# ### Data Cleaning


# %%
def clean_cpp_code(code) -> str:
    """
    - Removes single-line comments
    - Removes multi-line comments
    - Removes empty lines and leading/trailing whitespace from lines.
    """
    if not isinstance(code, str):
        raise ValueError("Input code must be a string")

    # Remove single-line comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

    # Remove multi-line comments
    code = re.sub(r"//.*", "", code)

    # Remove leading and trailing whitespace from each line
    lines = [line.strip() for line in code.splitlines() if line.strip()]

    cleaned_code = "\n".join(lines)
    return cleaned_code


if not df.empty and "raw_code" in df.columns:
    df["cleaned_code"] = df["raw_code"].apply(clean_cpp_code)

    if len(df) > 0:
        logger.info("--- Cleaning Example ---")
        idx_to_check = 0  # or a random index
        logger.info(f"Problem ID: {df['problem_id'].iloc[idx_to_check]}")
        logger.info("Original Code (first 500 chars):")
        print(df["raw_code"].iloc[idx_to_check][:500])
        logger.info("Cleaned Code (first 500 chars):")
        print(df["cleaned_code"].iloc[idx_to_check][:500])
else:
    logger.warning(
        "DataFrame is empty or 'raw_code_content' column is missing. Skipping cleaning demonstration."
    )

# %% [markdown]
# ### Save Metadata file

# %%
if not df.empty:
    entries = ["problem_id", "cleaned_code", "tags"]

    existing_columns = [col for col in entries if col in df.columns]

    if "cleaned_code" not in df.columns and "raw_code" in df.columns:
        logger.warning(
            "'cleaned_code' not found, metadata might not be fully processed for next steps."
        )

    df_metadata = df[existing_columns]

    try:
        df_metadata.to_csv(METADATA_FILE, index=False)
        logger.success(
            f"Successfully saved metadata for {len(df_metadata)} records to {METADATA_FILE}"
        )
        logger.info("Sample of metadata.csv content:")
        print(df_metadata.head())
    except Exception as e:
        logger.error(f"Error saving metadata file: {e}")

else:
    logger.warning("DataFrame is empty. No metadata file created.")

# %% [markdown]
# ## Lexer Tokenization

# %% [markdown]
# ### Configuration

# %%
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
LEXER_TOKENIZED_FILE = os.path.join(
    PROCESSED_DATA_DIR, "lexer_tokenized_data.pkl"
)


# %% [markdown]
# ### Lexer Tokenization


# %%
def tokenize_cpp(cleaned_code):
    """
    Tokenizes cleaned C++ code using Pygments CppLexer.
    Filters out comments, whitespace tokens.
    Returns a list of tuples (token type: str, token value).
    """
    if not isinstance(cleaned_code, str) or not cleaned_code.strip():
        raise ValueError("Input code must be a non-empty string")

    lexer = CppLexer()
    tokens = []

    try:
        for ttype, value in lexer.get_tokens(cleaned_code):
            if ttype in Comment or ttype in Text.Whitespace:
                continue
            tokens.append((str(ttype), value))
    except Exception as e:
        logger.error(
            f"Error during tokenization of code snippet: '{cleaned_code[:100]}...'. Error: {e}"
        )
        return []

    return tokens


# %%
if not df.empty and "cleaned_code" in df.columns:
    logger.info("Applying Lexer Tokenization to 'cleaned_code'...")

    df["cleaned_code"] = df["cleaned_code"].fillna("")
    df["lexer_tokens"] = df["cleaned_code"].apply(tokenize_cpp)

    logger.info("Lexer Tokenization completed.")
    logger.info(
        "Sample of data with lexer tokens (first 3 records, first 10 tokens):"
    )

    for idx, row in df.head(3).iterrows():
        logger.info(
            f"Problem ID: {row.get('problem_id', 'N/A')} - Tokens (first 10): {row.get('lexer_tokens', [])[:10]}"
        )

    try:
        df.to_pickle(LEXER_TOKENIZED_FILE)
        logger.success(f"Lexer tokenized data saved to {LEXER_TOKENIZED_FILE}")
    except Exception as e:
        logger.error(
            f"Error saving dataFrame to pickle '{LEXER_TOKENIZED_FILE}': {e}"
        )

elif df.empty:
    logger.warning("DataFrame is empty. No tokenization performed.")

elif "cleaned_code" not in df.columns:
    logger.warning(
        "'cleaned_code' column is missing. Cannot perform tokenization."
    )

# %% [markdown]
# ## Variable/Function Name Anonymization

# %% [markdown]
# ### Configuration

# %%
ANONYMIZED_TOKENIZED_FILE = os.path.join(
    PROCESSED_DATA_DIR, "anonymized_tokenized_data.pkl"
)

CPP_KEYWORDS = set(
    [
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "atomic_cancel",
        "atomic_commit",
        "atomic_noexcept",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char8_t",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "concept",
        "const",
        "consteval",
        "constexpr",
        "constinit",
        "const_cast",
        "continue",
        "co_await",
        "co_return",
        "co_yield",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "protected",
        "public",
        "reflexpr",
        "register",
        "reinterpret_cast",
        "requires",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "synchronized",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq",
        "std",
        "cin",
        "cout",
        "endl",
        "vector",
        "string",
        "map",
        "set",
        "pair",
        "iostream",
        "bits/stdc++.h",  # Heuristic
    ]
)


# %%
def anonymize_tokens(tokens):
    """
    Anonymizes identifiers and replaces literals in a list of (token_type, token_value) tuples.
    - Identifiers (Token.Name.*, not keywords) are replaced with <VAR_i> or <FUNC_i>.
    - Numbers (Token.Literal.Number.*) are replaced with <NUM>.
    - Strings/Chars (Token.Literal.String.*) are replaced with <STR>.
    Returns a list of processed tokens values.
    """
    if not isinstance(tokens, list):
        logger.warning(
            "anonymize_and_replace_literals received non-list input."
        )
        return []

    processed_tokens = []
    identifier_map = {}
    var_cnt, func_cnt = 0, 0

    for i, (ttype, value) in enumerate(tokens):
        nxt_value = tokens[i + 1][1] if i + 1 < len(tokens) else None

        if "Token.Literal.Number" in ttype:
            processed_tokens.append("<NUM>")
        elif "Token.Literal.String" in ttype:
            processed_tokens.append("<STR>")
        elif "Token.Name" in ttype:
            if value in CPP_KEYWORDS:
                processed_tokens.append(value)
            else:
                if value not in identifier_map:
                    is_heuristic = (
                        (nxt_value == "(")
                        or ("Function" in ttype)
                        or ("Class" in ttype)
                    )
                    if is_heuristic:
                        identifier_map[value] = f"<FUNC_{func_cnt}>"
                        func_cnt += 1
                    else:
                        identifier_map[value] = f"<VAR_{var_cnt}>"
                        var_cnt += 1
                processed_tokens.append(identifier_map[value])
        else:
            processed_tokens.append(value)

    return processed_tokens


# %% [markdown]
# ### Apply Anonymization

# %%
assert not df.empty, "DataFrame should not be empty before anonymization."
assert (
    "lexer_tokens" in df.columns
), "'lexer_tokens' column should exist before anonymization."

logger.info("Applying anonymization and literal replacement...")

df["anonymized_tokens"] = df["lexer_tokens"].apply(anonymize_tokens)

logger.info("Anonymization and literal replacement completed.")
logger.info(
    "Sample of data with anonymized token values (first 3 records, first 15 tokens):"
)

for index, row in df.head(3).iterrows():
    logger.info(f"Problem ID: {row.get('problem_id', 'N/A')}")
    logger.info(
        f"Original Tokens (first 15): {row.get('lexer_tokens', [])[:15]}"
    )
    logger.info(
        f"Anonymized Tokens (first 15): {row.get('anonymized_tokens', [])[:15]}"
    )
    logger.info("-" * 30)

try:
    df.to_pickle(ANONYMIZED_TOKENIZED_FILE)
    logger.success(
        f"Anonymized tokenized data saved to {ANONYMIZED_TOKENIZED_FILE}"
    )
except Exception as e:
    logger.error(
        f"Error saving anonymized tokenized data to pickle '{ANONYMIZED_TOKENIZED_FILE}': {e}"
    )

# %% [markdown]
# ### Tag Processing (One-Hot Encoding)

# %% [markdown]
# #### Configuration

# %%
FINAL_PROCESSED_FILE = os.path.join(
    PROCESSED_DATA_DIR, "final_processed_data.pkl"
)
MLB_FILE = os.path.join(PROCESSED_DATA_DIR, "mlb_encoder.pkl")

# %%
if not df.empty and "tags" in df.columns:
    logger.info(
        f"DataFrame sample before One-Hot Encoding (first {min(3, len(df))} rows):"
    )

    for i in range(min(3, len(df))):
        logger.info(
            f"Row {i} Problem ID: {df['problem_id'].iloc[i]} - Tags List: {df['tags'].iloc[i]}"
        )

    if not df["tags"].empty and isinstance(df["tags"].iloc[0], str):
        logger.warning(
            "Tags_list appears to be strings, attempting ast.literal_eval. Ensure this column was correctly processed earlier."
        )
        try:
            df["tags"] = df["tags"].apply(ast.literal_eval)
        except Exception as e:
            logger.error(f"Error converting 'tags' column to list: {e}")
            df = pd.DataFrame()

else:
    logger.warning(
        "DataFrame is empty or 'tags_list' column is missing. Cannot proceed with One-Hot Encoding."
    )

# %% [markdown]
# ### Apply MultiLabelBinarizer

# %%
if not df.empty and "tags" in df.columns:
    logger.info("Applying MultiLabelBinarizer to 'tags_list'...")

    mlb = MultiLabelBinarizer()

    try:
        df["tags_for_mlb"] = df["tags"].apply(
            lambda x: x if isinstance(x, list) else []
        )

        one_hot_encoded = mlb.fit_transform(df["tags_for_mlb"])
        df["one_hot_tags"] = [list(row) for row in one_hot_encoded]

        num_unique_tags = len(mlb.classes_)
        logger.info(
            f"One-Hot Encoding completed. Number of unique tags: {num_unique_tags}"
        )
        logger.info("Sample of One-Hot Encoded data (first 3 records):")

        for idx, row in df.head(3).iterrows():
            logger.info(f"Problem ID: {row.get('problem_id', 'N/A')}")
            logger.info(f"Original Tags: {row.get('tags', [])}")
            logger.info(f"One-Hot Labels: {row.get('one_hot_tags', [])}")
            logger.info("-" * 30)

        try:
            joblib.dump(mlb, MLB_FILE)
            logger.success(f"MultiLabelBinarizer saved to {MLB_FILE}")
        except Exception as e:
            logger.error(
                f"Error saving MultiLabelBinarizer to file '{MLB_FILE}': {e}"
            )

        if "tags_for_mlb" in df.columns:
            df_to_save = df.drop(columns=["tags_for_mlb"])
        else:
            df_to_save = df

        try:
            df_to_save.to_pickle(FINAL_PROCESSED_FILE)
            logger.success(
                f"Final processed data saved to {FINAL_PROCESSED_FILE}"
            )
        except Exception as e:
            logger.error(
                f"Error saving final processed data to pickle '{FINAL_PROCESSED_FILE}': {e}"
            )

    except Exception as e:
        logger.error(f"An error occurred during MultiLabelBinarization: {e}")
        logger.error(
            "Please check the format of the 'tags' column. It should be a Series of lists of strings."
        )

else:
    logger.warning(
        "DataFrame is empty or 'tags' column is missing. Skipping One-Hot Encoding."
    )

# %%
import matplotlib.pyplot as plt
import seaborn as sns

if "df" not in locals() or df.empty or "anonymized_tokens" not in df.columns:
    if os.path.exists(FINAL_PROCESSED_FILE):
        logger.info(
            f"Loading DataFrame with anonymized tokens from {FINAL_PROCESSED_FILE}..."
        )
        df = pd.read_pickle(FINAL_PROCESSED_FILE)
        logger.info(f"Successfully loaded data with {len(df)} records.")
    else:
        logger.error(
            f"File {FINAL_PROCESSED_FILE} not found. Cannot analyze sequence lengths."
        )
        df = pd.DataFrame()
else:
    logger.info(
        "DataFrame 'df' with 'anonymized_tokens' already exists. Proceeding with length analysis."
    )

if not df.empty and "anonymized_tokens" in df.columns:
    # Calculate the length of each token sequence
    df["token_sequence_length"] = df["anonymized_tokens"].apply(len)

    logger.info("\n--- Sequence Length Statistics ---")
    logger.info(f"Min length: {df['token_sequence_length'].min()}")
    logger.info(f"Max length: {df['token_sequence_length'].max()}")
    logger.info(f"Mean length: {df['token_sequence_length'].mean():.2f}")
    logger.info(f"Median length: {df['token_sequence_length'].median()}")

    # --- Percentile Analysis ---
    percentiles_to_check = [
        0.50,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.98,
        0.99,
        1.00,
    ]
    logger.info("\n--- Percentile Distribution of Sequence Lengths ---")
    for p in percentiles_to_check:
        logger.info(
            f"{p*100:.0f}th percentile: {df['token_sequence_length'].quantile(p):.0f} tokens"
        )

    # --- Plotting the distribution (optional but highly recommended) ---
    plt.figure(figsize=(12, 6))
    sns.histplot(df["token_sequence_length"], bins=50, kde=False)
    plt.title("Distribution of Token Sequence Lengths")
    plt.xlabel("Sequence Length (Number of Tokens)")
    plt.ylabel("Frequency")
    plt.grid(True)
    # You might want to set x-axis limits if there are extreme outliers making the plot hard to read
    # For example, limit to the 99th percentile to see the main distribution better
    # plt.xlim(0, df['token_sequence_length'].quantile(0.99) * 1.1) # Add 10% margin
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df["token_sequence_length"])
    plt.title("Box Plot of Token Sequence Lengths")
    plt.xlabel("Sequence Length (Number of Tokens)")
    plt.grid(True)
    plt.show()

else:
    logger.warning(
        "DataFrame is empty or 'anonymized_token' column is missing. Cannot analyze sequence lengths."
    )

# %% [markdown]
# ## Final Dataset Preparation (for CodeBERT ...)

# %% [markdown]
# #### Configuration

# %%
MODEL_INPUT_DATA_FILE_CODEBERT = os.path.join(
    PROCESSED_DATA_DIR, "model_input_data_codebert.npz"
)
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
MAX_SEQUENCE_LENGTH = 512

TEST_SET_SIZE = 0.15
VALIDATION_SET_SIZE = 0.15
RANDOM_STATE = 42  # For reproducibility of splits

# %%
if (
    not df.empty
    and "anonymized_tokens" in df.columns
    and "one_hot_tags" in df.columns
):
    logger.info(
        f"DataFrame sample before CodeBERT tokenization (first {min(3, len(df))} rows):"
    )
    for i in range(min(3, len(df))):
        logger.info(f"Row {i} Problem ID: {df['problem_id'].iloc[i]}")
        logger.info(
            f"Anonymized Tokens (first 10): {df['anonymized_tokens'].iloc[i][:10]}"
        )
        logger.info(
            f"One-Hot Labels (first 10 elements): {df['one_hot_tags'].iloc[i][:10]}"
        )
else:
    logger.warning(
        "DataFrame is empty or required columns ('anonymized_token_values', 'one_hot_labels') are missing."
    )

# %% [markdown]
# ### Convert Token Lists to Strings and Tokenize with CodeBERT Tokenizer

# %%
if not df.empty and "anonymized_tokens" in df.columns:
    logger.info(
        "Converting anonymized token lists to space-separated strings..."
    )
    df["anonymized_code_bert"] = df["anonymized_tokens"].apply(
        lambda x: " ".join(x)
    )

    logger.info("Initializing CodeBERT tokenizer...")
    try:
        codebert_tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
        logger.info(
            f"CodeBERT tokenizer '{CODEBERT_MODEL_NAME}' loaded successfully."
        )
    except Exception as e:
        logger.error(f"Failed to load CodeBERT tokenizer: {e}")
        codebert_tokenizer = None

    if codebert_tokenizer:
        logger.info(
            f"Tokenizing 'anonymized_code_string_for_bert' with CodeBERT tokenizer (max_length={MAX_SEQUENCE_LENGTH})..."
        )

        all_input_ids = []
        all_attention_masks = []
        sequences_to_tokenize = df["anonymized_code_bert"].tolist()

        for i, code in enumerate(sequences_to_tokenize):
            if i % 100 == 0 and i > 0:
                logger.info(
                    f"Tokenizing sample {i} / {len(sequences_to_tokenize)}..."
                )

            try:
                encoded_dict = codebert_tokenizer.encode_plus(
                    code,
                    add_special_tokens=True,
                    max_length=MAX_SEQUENCE_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="np",
                )
                all_input_ids.append(encoded_dict["input_ids"][0])
                all_attention_masks.append(encoded_dict["attention_mask"][0])

            except Exception as e:
                logger.error(
                    f"Error tokenizing code for problem_id {df['problem_id'].iloc[i]}: {e}"
                )
                # Placeholders for failed tokenization
                all_input_ids.append(np.zeros(MAX_SEQUENCE_LENGTH, dtype=int))
                all_attention_masks.append(
                    np.zeros(MAX_SEQUENCE_LENGTH, dtype=int)
                )

        X_input_ids = np.array(all_input_ids)
        X_attention_mask = np.array(all_attention_masks)

        try:
            y_labels = np.array(df["one_hot_tags"].tolist(), dtype=np.int32)
            logger.info(f"Shape of X_input_ids: {X_input_ids.shape}")
            logger.info(f"Shape of X_attention_mask: {X_attention_mask.shape}")
            logger.info(f"Shape of y_labels: {y_labels.shape}")

            if X_input_ids.shape[0] != y_labels.shape[0]:
                logger.error(
                    "Mismatch between number of input IDs and labels. Check your data."
                )
            else:
                logger.info("CodeBERT tokenization complete and shapes match.")

        except Exception as e:
            logger.error(
                f"Error converting 'one_hot_labels' to NumPy array or shape mismatch: {e}"
            )
            # Set to None to prevent saving bad data
            X_input_ids, X_attention_mask, y_labels = None, None, None

else:
    logger.warning(
        "DataFrame is empty or 'anonymized_token' column is missing. Skipping CodeBERT tokenization."
    )
    X_input_ids, X_attention_mask, y_labels = None, None, None

# %% [markdown]
# #### Split Data into Training, Validation, and Test Sets

# %%
if (
    X_input_ids is not None
    and X_attention_mask is not None
    and y_labels is not None
):
    logger.info("Splitting data into training, validation, and test sets...")

    # Separate out the test set
    (
        train_val_input_ids,
        X_test_input_ids,
        train_val_attention_mask,
        X_test_attention_mask,
        train_val_labels,
        y_test,
    ) = train_test_split(
        X_input_ids,
        X_attention_mask,
        y_labels,
        test_size=TEST_SET_SIZE,
        random_state=RANDOM_STATE,
        stratify=None,
    )

    relative_val_size = VALIDATION_SET_SIZE / (1 - TEST_SET_SIZE)

    (
        X_train_input_ids,
        X_val_input_ids,
        X_train_attention_mask,
        X_val_attention_mask,
        y_train,
        y_val,
    ) = train_test_split(
        train_val_input_ids,
        train_val_attention_mask,
        train_val_labels,
        test_size=relative_val_size,
        random_state=RANDOM_STATE,
        stratify=None,
    )

    logger.info(
        f"Training set shapes: IDs {X_train_input_ids.shape}, Masks {X_train_attention_mask.shape}, Labels {y_train.shape}"
    )
    logger.info(
        f"Validation set shapes: IDs {X_val_input_ids.shape}, Masks {X_val_attention_mask.shape}, Labels {y_val.shape}"
    )
    logger.info(
        f"Test set shapes: IDs {X_test_input_ids.shape}, Masks {X_test_attention_mask.shape}, Labels {y_test.shape}"
    )

    try:
        np.savez_compressed(
            MODEL_INPUT_DATA_FILE_CODEBERT,
            X_train_input_ids=X_train_input_ids,
            X_train_attention_mask=X_train_attention_mask,
            y_train=y_train,
            X_val_input_ids=X_val_input_ids,
            X_val_attention_mask=X_val_attention_mask,
            y_val=y_val,
            X_test_input_ids=X_test_input_ids,
            X_test_attention_mask=X_test_attention_mask,
            y_test=y_test,
        )
        logger.success(
            f"Model input datasets saved to: {MODEL_INPUT_DATA_FILE_CODEBERT}"
        )

    except Exception as e:
        logger.error(
            f"Error saving model input data to '{MODEL_INPUT_DATA_FILE_CODEBERT}': {e}"
        )

else:
    logger.error(
        "Tokenized inputs (X_input_ids, X_attention_mask) or labels (y_labels) are None. Cannot split or save data."
    )

# %% [markdown]
# ## Model Training

# %% [markdown]
# ### Setup

# %%
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# %%
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LEARNING_RATE = 5e-5
EPOCHS = 10
BATCH_SIZE = 16
BEST_MODEL_PATH = os.path.join(
    MODEL_DIR, "codebert_multi_label_classifier_best_tf_format"
)

# %%
if os.path.exists(MLB_FILE):
    mlb = joblib.load(MLB_FILE)
    num_labels = len(mlb.classes_)
    logger.info(
        f"Loaded MultiLabelBinarizer. Number of unique labels: {num_labels}"
    )
    logger.info(f"Classes: {mlb.classes_}")
else:
    logger.error(
        f"MultiLabelBinarizer file not found at {MLB_FILE}. Cannot determine number of labels."
    )
    num_labels = -1

# %% [markdown]
# ### Load the preprocessed Dataset

# %%
if num_labels > 0 and os.path.exists(MODEL_INPUT_DATA_FILE_CODEBERT):
    logger.info(
        f"Loading model input datasets from {MODEL_INPUT_DATA_FILE_CODEBERT}..."
    )
    loaded_data = np.load(MODEL_INPUT_DATA_FILE_CODEBERT)

    X_train_input_ids = loaded_data["X_train_input_ids"]
    X_train_attention_mask = loaded_data["X_train_attention_mask"]
    y_train = loaded_data["y_train"]

    X_val_input_ids = loaded_data["X_val_input_ids"]
    X_val_attention_mask = loaded_data["X_val_attention_mask"]
    y_val = loaded_data["y_val"]

    logger.info("Datasets loaded successfully.")
    logger.info(
        f"Training set shapes: IDs {X_train_input_ids.shape}, Masks {X_train_attention_mask.shape}, Labels {y_train.shape}"
    )
    logger.info(
        f"Validation set shapes: IDs {X_val_input_ids.shape}, Masks {X_val_attention_mask.shape}, Labels {y_val.shape}"
    )

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                {
                    "input_ids": X_train_input_ids,
                    "attention_mask": X_train_attention_mask,
                },
                y_train,
            )
        )
        .shuffle(len(X_train_input_ids))
        .batch(BATCH_SIZE)
    )

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "input_ids": X_val_input_ids,
                "attention_mask": X_val_attention_mask,
            },
            y_val,
        )
    ).batch(BATCH_SIZE)

    logger.info("TensorFlow Datasets prepared.")

else:
    logger.error(
        "Could not load datasets or num_labels is invalid. Please check previous steps."
    )
    train_dataset, val_dataset = None, None
    X_train_input_ids = np.array([])

# %% [markdown]
# ### Build/Compile the CodeBERT Model

# %%
if num_labels > 0 and train_dataset is not None:
    logger.info(
        f"Building TFAutoModelForSequenceClassification with CodeBERT: '{CODEBERT_MODEL_NAME}'..."
    )

    try:

        model = TFAutoModelForSequenceClassification.from_pretrained(
            CODEBERT_MODEL_NAME,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        for i, layer in enumerate(model.roberta.encoder.layer):
            if i < 10:
                layer.trainable = False

        if "y_train" in locals():
            class_weights = []
            total_samples = len(y_train)
            for i in range(num_labels):
                pos_samples = np.sum(y_train[:, i])
                if pos_samples > 0:
                    weight = total_samples / (2 * pos_samples)
                else:
                    weight = 1.0
                class_weights.append(weight)

            def weighted_binary_crossentropy(y_true, y_pred):
                weights = tf.constant(class_weights, dtype=tf.float32)
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)

                bce = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y_true, logits=y_pred
                )
                weighted_bce = bce * weights
                return tf.reduce_mean(weighted_bce)

            loss_fn = weighted_binary_crossentropy
        else:
            assert False
            loss_fn = BinaryCrossentropy(from_logits=True)

        optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

        metrics_list = [
            AUC(
                name="auc_pr",
                curve="PR",
                multi_label=True,
                num_labels=num_labels,
            ),  # Precision-Recall AUC
            Precision(name="precision", thresholds=0.5),
            BinaryAccuracy(name="binary_accuracy"),
        ]

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_list)

        model.summary()
        logger.info("CodeBERT model built and compiled successfully.")

    except Exception as e:
        logger.error(
            f"Error building TFAutoModelForSequenceClassification: {e}"
        )
        model = None

else:
    logger.error(
        "Cannot build model due to issues in data loading or num_labels."
    )
    model = None

# %% [markdown]
# ### Train the Model

# %%
if (
    model is not None
    and train_dataset is not None
    and val_dataset is not None
    and len(X_train_input_ids) > 0
):
    logger.info("Starting model training...")

    early_stopping = EarlyStopping(
        monitor="val_precision",
        mode="max",
        patience=3,
        restore_best_weights=True,
    )

    model_checkpoint = ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_precision",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        save_format="tf",
    )

    callbacks_list = [early_stopping, model_checkpoint]

    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks_list,
        )

        logger.success(
            f"Model training finished. Best model saved to {BEST_MODEL_PATH} (if validation improved)."
        )

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")

else:
    logger.error("Model or datasets are not available. Skipping training.")


# %% [markdown]
# ## Model Evaluation

# %%
from sklearn.metrics import (
    jaccard_score,
    f1_score,
    multilabel_confusion_matrix,
    classification_report,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import tensorflow as tf

# %% [markdown]
# ### Load Best Model and Test Data

# %%
if os.path.exists(BEST_MODEL_PATH):
    logger.info(f"Loading best model from {BEST_MODEL_PATH}...")
    try:
        best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
        logger.success("Best model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading best model: {e}")
        best_model = None
else:
    logger.warning(
        f"Best model file not found at {BEST_MODEL_PATH}. Using current model for evaluation."
    )
    best_model = model if "model" in locals() else None

# Load test data
if os.path.exists(MODEL_INPUT_DATA_FILE_CODEBERT):
    logger.info("Loading test dataset...")
    loaded_data = np.load(MODEL_INPUT_DATA_FILE_CODEBERT)

    X_test_input_ids = loaded_data["X_test_input_ids"]
    X_test_attention_mask = loaded_data["X_test_attention_mask"]
    y_test = loaded_data["y_test"]

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "input_ids": X_test_input_ids,
                "attention_mask": X_test_attention_mask,
            },
            y_test,
        )
    ).batch(BATCH_SIZE)

    logger.info(f"Test dataset loaded. Shape: {X_test_input_ids.shape}")
else:
    logger.error("Test dataset file not found. Cannot perform evaluation.")
    test_dataset = None

# %% [markdown]
# ### Make Predictions on Test Set

# %%
if best_model is not None and test_dataset is not None:
    logger.info("Making predictions on test set...")

    # Get predictions
    try:
        y_pred_logits = best_model.predict(test_dataset)

        # Convert logits to probabilities using sigmoid
        y_pred_proba = tf.nn.sigmoid(y_pred_logits).numpy()

        # Convert probabilities to binary predictions (threshold = 0.5)
        y_pred_binary = (y_pred_proba > 0.5).astype(int)

        logger.info(f"Predictions completed. Shape: {y_pred_proba.shape}")
        logger.info(f"Binary predictions shape: {y_pred_binary.shape}")

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        y_pred_proba = None
        y_pred_binary = None
else:
    logger.error("Model or test dataset not available. Skipping prediction.")
    y_pred_proba = None
    y_pred_binary = None

# %% [markdown]
# ### Calculate Evaluation Metrics

# %%
if y_pred_binary is not None and y_test is not None:
    logger.info("Calculating evaluation metrics...")

    try:
        # Jaccard Score (IoU)
        jaccard_micro = jaccard_score(y_test, y_pred_binary, average="micro")
        jaccard_macro = jaccard_score(y_test, y_pred_binary, average="macro")
        jaccard_weighted = jaccard_score(
            y_test, y_pred_binary, average="weighted"
        )

        # F1 Scores
        f1_micro = f1_score(y_test, y_pred_binary, average="micro")
        f1_macro = f1_score(y_test, y_pred_binary, average="macro")
        f1_weighted = f1_score(y_test, y_pred_binary, average="weighted")

        # ROC AUC Score
        try:
            roc_auc_micro = roc_auc_score(y_test, y_pred_proba, average="micro")
            roc_auc_macro = roc_auc_score(y_test, y_pred_proba, average="macro")
            roc_auc_weighted = roc_auc_score(
                y_test, y_pred_proba, average="weighted"
            )
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            roc_auc_micro = roc_auc_macro = roc_auc_weighted = None

        # Average Precision Score
        try:
            avg_precision_micro = average_precision_score(
                y_test, y_pred_proba, average="micro"
            )
            avg_precision_macro = average_precision_score(
                y_test, y_pred_proba, average="macro"
            )
            avg_precision_weighted = average_precision_score(
                y_test, y_pred_proba, average="weighted"
            )
        except Exception as e:
            logger.warning(f"Could not calculate Average Precision: {e}")
            avg_precision_micro = avg_precision_macro = (
                avg_precision_weighted
            ) = None

        # Print evaluation results
        logger.info("\n" + "=" * 60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("=" * 60)

        logger.info(f"\nJaccard Score (IoU):")
        logger.info(f"  - Micro Average: {jaccard_micro:.4f}")
        logger.info(f"  - Macro Average: {jaccard_macro:.4f}")
        logger.info(f"  - Weighted Average: {jaccard_weighted:.4f}")

        logger.info(f"\nF1 Score:")
        logger.info(f"  - Micro Average: {f1_micro:.4f}")
        logger.info(f"  - Macro Average: {f1_macro:.4f}")
        logger.info(f"  - Weighted Average: {f1_weighted:.4f}")

        if roc_auc_micro is not None:
            logger.info(f"\nROC AUC Score:")
            logger.info(f"  - Micro Average: {roc_auc_micro:.4f}")
            logger.info(f"  - Macro Average: {roc_auc_macro:.4f}")
            logger.info(f"  - Weighted Average: {roc_auc_weighted:.4f}")

        if avg_precision_micro is not None:
            logger.info(f"\nAverage Precision Score:")
            logger.info(f"  - Micro Average: {avg_precision_micro:.4f}")
            logger.info(f"  - Macro Average: {avg_precision_macro:.4f}")
            logger.info(f"  - Weighted Average: {avg_precision_weighted:.4f}")

    except Exception as e:
        logger.error(f"Error calculating evaluation metrics: {e}")

# %% [markdown]
# ### Per-Label Analysis

# %%
if y_pred_binary is not None and y_test is not None and "mlb" in locals():
    logger.info("\nCalculating per-label metrics...")

    try:
        # Calculate per-label F1 scores
        f1_per_label = f1_score(y_test, y_pred_binary, average=None)

        # Calculate per-label precision and recall
        from sklearn.metrics import precision_score, recall_score

        precision_per_label = precision_score(
            y_test, y_pred_binary, average=None, zero_division=0
        )
        recall_per_label = recall_score(
            y_test, y_pred_binary, average=None, zero_division=0
        )

        # Create a DataFrame for better visualization
        per_label_metrics = pd.DataFrame(
            {
                "Tag": mlb.classes_,
                "F1_Score": f1_per_label,
                "Precision": precision_per_label,
                "Recall": recall_per_label,
                "Support": y_test.sum(
                    axis=0
                ),  # Number of true positives for each label
            }
        )

        # Sort by F1 score for better readability
        per_label_metrics = per_label_metrics.sort_values(
            "F1_Score", ascending=False
        )

        logger.info("\nTop 10 Tags by F1 Score:")
        logger.info(per_label_metrics.head(10).to_string(index=False))

        logger.info("\nBottom 10 Tags by F1 Score:")
        logger.info(per_label_metrics.tail(10).to_string(index=False))

        # Save detailed results
        results_file = os.path.join(
            PROCESSED_DATA_DIR, "evaluation_results.csv"
        )
        per_label_metrics.to_csv(results_file, index=False)
        logger.info(f"Detailed per-label results saved to {results_file}")

    except Exception as e:
        logger.error(f"Error in per-label analysis: {e}")

# %% [markdown]
# ### Visualization of Results

# %%
if y_pred_binary is not None and y_test is not None:
    # 1. Confusion Matrix Heatmap for top tags
    try:
        # Select top 10 most frequent tags for confusion matrix visualization
        if "per_label_metrics" in locals():
            top_tags_indices = per_label_metrics.nlargest(10, "Support").index[
                :10
            ]
            top_tag_names = [mlb.classes_[i] for i in top_tags_indices]

            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()

            for idx, tag_idx in enumerate(top_tags_indices):
                cm = multilabel_confusion_matrix(y_test, y_pred_binary)[tag_idx]
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx])
                axes[idx].set_title(f"Tag: {mlb.classes_[tag_idx]}")
                axes[idx].set_xlabel("Predicted")
                axes[idx].set_ylabel("Actual")

            plt.tight_layout()
            plt.suptitle(
                "Confusion Matrices for Top 10 Most Frequent Tags", y=1.02
            )
            plt.show()

    except Exception as e:
        logger.warning(f"Could not create confusion matrix visualization: {e}")

    # 2. F1 Score Distribution
    try:
        if "f1_per_label" in locals():
            plt.figure(figsize=(12, 6))
            plt.hist(
                f1_per_label,
                bins=30,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )
            plt.axvline(
                f1_per_label.mean(),
                color="red",
                linestyle="--",
                label=f"Mean F1: {f1_per_label.mean():.3f}",
            )
            plt.xlabel("F1 Score")
            plt.ylabel("Number of Tags")
            plt.title("Distribution of F1 Scores Across All Tags")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

    except Exception as e:
        logger.warning(f"Could not create F1 score distribution plot: {e}")

    # 3. Prediction Probability Distribution
    try:
        plt.figure(figsize=(12, 6))
        plt.hist(
            y_pred_proba.flatten(),
            bins=50,
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
        )
        plt.axvline(
            0.5, color="red", linestyle="--", label="Decision Threshold (0.5)"
        )
        plt.xlabel("Prediction Probability")
        plt.ylabel("Frequency")
        plt.title("Distribution of Prediction Probabilities")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        logger.warning(f"Could not create probability distribution plot: {e}")

    # 4. Support vs Performance Analysis
    try:
        if "per_label_metrics" in locals():
            plt.figure(figsize=(12, 8))

            # Create scatter plot
            scatter = plt.scatter(
                per_label_metrics["Support"],
                per_label_metrics["F1_Score"],
                alpha=0.6,
                s=60,
                c=per_label_metrics["F1_Score"],
                cmap="viridis",
            )

            plt.xlabel("Support (Number of True Instances)")
            plt.ylabel("F1 Score")
            plt.title("F1 Score vs Support for Each Tag")
            plt.colorbar(scatter, label="F1 Score")
            plt.grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(
                per_label_metrics["Support"], per_label_metrics["F1_Score"], 1
            )
            p = np.poly1d(z)
            plt.plot(
                per_label_metrics["Support"],
                p(per_label_metrics["Support"]),
                "r--",
                alpha=0.8,
                label="Trend Line",
            )
            plt.legend()
            plt.show()

    except Exception as e:
        logger.warning(f"Could not create support vs performance plot: {e}")

# %% [markdown]
# ### Sample Predictions Analysis

# %%
if y_pred_binary is not None and y_test is not None and "mlb" in locals():
    logger.info("\nAnalyzing sample predictions...")

    try:
        # Select a few samples for detailed analysis
        sample_indices = np.random.choice(
            len(y_test), size=min(5, len(y_test)), replace=False
        )

        for i, idx in enumerate(sample_indices):
            logger.info(f"\n--- Sample {i+1} (Index: {idx}) ---")

            # True labels
            true_labels = [
                mlb.classes_[j]
                for j in range(len(mlb.classes_))
                if y_test[idx][j] == 1
            ]

            # Predicted labels
            pred_labels = [
                mlb.classes_[j]
                for j in range(len(mlb.classes_))
                if y_pred_binary[idx][j] == 1
            ]

            # Top predicted probabilities
            top_prob_indices = np.argsort(y_pred_proba[idx])[::-1][:10]
            top_probs = [
                (mlb.classes_[j], y_pred_proba[idx][j])
                for j in top_prob_indices
            ]

            logger.info(f"True Labels: {true_labels}")
            logger.info(f"Predicted Labels: {pred_labels}")
            logger.info(f"Top 10 Prediction Probabilities:")
            for label, prob in top_probs:
                logger.info(f"  {label}: {prob:.4f}")

            # Calculate sample-level metrics
            sample_intersection = set(true_labels) & set(pred_labels)
            sample_union = set(true_labels) | set(pred_labels)
            sample_jaccard = (
                len(sample_intersection) / len(sample_union)
                if sample_union
                else 0
            )

            logger.info(f"Sample Jaccard Score: {sample_jaccard:.4f}")
            logger.info(f"Correct Predictions: {len(sample_intersection)}")
            logger.info(f"Total True Labels: {len(true_labels)}")
            logger.info(f"Total Predicted Labels: {len(pred_labels)}")

    except Exception as e:
        logger.error(f"Error in sample predictions analysis: {e}")

# %% [markdown]
# ### Model Performance Summary

# %%
logger.info("\n" + "=" * 60)
logger.info("MODEL PERFORMANCE SUMMARY")
logger.info("=" * 60)

summary_metrics = {
    "Dataset Size": len(y_test) if y_test is not None else "N/A",
    "Number of Labels": len(mlb.classes_) if "mlb" in locals() else "N/A",
    "Average Labels per Sample": (
        np.mean(y_test.sum(axis=1)) if y_test is not None else "N/A"
    ),
}

if "jaccard_micro" in locals():
    summary_metrics.update(
        {
            "Jaccard Score (Micro)": f"{jaccard_micro:.4f}",
            "F1 Score (Micro)": f"{f1_micro:.4f}",
            "F1 Score (Macro)": f"{f1_macro:.4f}",
        }
    )

if "roc_auc_micro" in locals() and roc_auc_micro is not None:
    summary_metrics.update(
        {
            "ROC AUC (Micro)": f"{roc_auc_micro:.4f}",
            "ROC AUC (Macro)": f"{roc_auc_macro:.4f}",
        }
    )

for key, value in summary_metrics.items():
    logger.info(f"{key}: {value}")

logger.info("=" * 60)

# Save evaluation summary
if "summary_metrics" in locals():
    summary_file = os.path.join(PROCESSED_DATA_DIR, "evaluation_summary.json")
    import json

    # Convert numpy types to native Python types for JSON serialization
    json_summary = {}
    for key, value in summary_metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            json_summary[key] = value.item()
        else:
            json_summary[key] = value

    try:
        with open(summary_file, "w") as f:
            json.dump(json_summary, f, indent=2)
        logger.info(f"Evaluation summary saved to {summary_file}")
    except Exception as e:
        logger.warning(f"Could not save evaluation summary: {e}")

logger.info("Model evaluation completed!")
