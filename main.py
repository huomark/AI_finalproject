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
ANONYMIZED_TOKENIZED_FILE = os.path.join(PROCESSED_DATA_DIR, "anonymized_tokenized_data.pkl")

CPP_KEYWORDS = set([
    "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
    "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case",
    "catch", "char", "char8_t", "char16_t", "char32_t", "class", "compl",
    "concept", "const", "consteval", "constexpr", "constinit", "const_cast",
    "continue", "co_await", "co_return", "co_yield", "decltype", "default",
    "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit",
    "export", "extern", "false", "float", "for", "friend", "goto", "if",
    "inline", "int", "long", "mutable", "namespace", "new", "noexcept",
    "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private",
    "protected", "public", "reflexpr", "register", "reinterpret_cast",
    "requires", "return", "short", "signed", "sizeof", "static",
    "static_assert", "static_cast", "struct", "switch", "synchronized",
    "template", "this", "thread_local", "throw", "true", "try", "typedef",
    "typeid", "typename", "union", "unsigned", "using", "virtual", "void",
    "volatile", "wchar_t", "while", "xor", "xor_eq",
    "std", "cin", "cout", "endl", "vector", "string", "map", "set", "pair", "iostream", "bits/stdc++.h"  # Heuristic
])


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
        logger.warning("anonymize_and_replace_literals received non-list input.")
        return []

    processed_tokens = []
    identifier_map = {}
    var_cnt, func_cnt = 0, 0

    for i, (ttype, value) in enumerate(tokens):
        nxt_value = tokens[i + 1][1] if i + 1 < len(tokens) else None

        if 'Token.Literal.Number' in ttype:
            processed_tokens.append("<NUM>")
        elif 'Token.Literal.String' in ttype:
            processed_tokens.append("<STR>")
        elif 'Token.Name' in ttype:
            if value in CPP_KEYWORDS:
                processed_tokens.append(value)
            else:
                if value not in identifier_map:
                    is_heuristic = (nxt_value == '(') or ('Function' in ttype) or ('Class' in ttype)
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
assert "lexer_tokens" in df.columns, "'lexer_tokens' column should exist before anonymization."

logger.info("Applying anonymization and literal replacement...")

df['anonymized_tokens'] = df['lexer_tokens'].apply(anonymize_tokens)

logger.info("Anonymization and literal replacement completed.")
logger.info("Sample of data with anonymized token values (first 3 records, first 15 tokens):")

for index, row in df.head(3).iterrows():
    logger.info(f"Problem ID: {row.get('problem_id', 'N/A')}")
    logger.info(f"Original Tokens (first 15): {row.get('lexer_tokens', [])[:15]}")
    logger.info(f"Anonymized Tokens (first 15): {row.get('anonymized_tokens', [])[:15]}")
    logger.info("-" * 30)

try:
    df.to_pickle(ANONYMIZED_TOKENIZED_FILE)
    logger.success(f"Anonymized tokenized data saved to {ANONYMIZED_TOKENIZED_FILE}")
except Exception as e:
    logger.error(f"Error saving anonymized tokenized data to pickle '{ANONYMIZED_TOKENIZED_FILE}': {e}")
