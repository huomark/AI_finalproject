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
    print(f"ERROR: Submissions directory not found at {SUBMISSIONS_DIR}")
if not os.path.exists(TAG_DIR):
    print(f"ERROR: Tags directory not found at {TAG_DIR}")


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
        print(f"ERROR: Code file {code_filepath} not found, but was listed")
        continue
    except Exception as e:
        print(f"ERROR: Could not read code file {code_filepath}: {e}")
        continue

    # Read tags
    try:
        with open(tag_filepath, "r", encoding="utf-8") as tag_file:
            tag_content = tag_file.read()
            tag_list = re.split(
                r"\s*,\s*", tag_content.strip()
            )  # Split by commas and strip whitespace
    except FileNotFoundError:
        print(f"ERROR: Tag file {tag_filepath} not found, but was listed")
        continue
    except Exception as e:
        print(f"ERROR: Could not read tag file {tag_filepath}: {e}")
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
    print(f"\nSuccessfully loaded {len(df)} records.")
    print("\nSample of loaded data:")
    print(df.head())
else:
    print(
        "\nNo data records were loaded. Please check your `SUBMISSIONS_DIR` and `TAGS_DIR` paths and content."
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
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL | re.MULTILINE)

    # Remove multi-line comments
    code = re.sub(r"//.*", "", code)

    # Remove leading and trailing whitespace from each line
    lines = [line.strip() for line in code.splitlines() if line.strip()]

    cleaned_code = "\n".join(lines)
    return cleaned_code


if not df.empty and "raw_code" in df.columns:
    df["cleaned_code"] = df["raw_code"].apply(clean_cpp_code)

    if len(df) > 0:
        print("\n--- Cleaning Example ---")
        idx_to_check = 0  # or a random index
        print(f"Problem ID: {df['problem_id'].iloc[idx_to_check]}")
        print("\nOriginal Code (first 500 chars):")
        print(df["raw_code"].iloc[idx_to_check][:500])
        print("\nCleaned Code (first 500 chars):")
        print(df["cleaned_code"].iloc[idx_to_check][:500])
else:
    print(
        "\nDataFrame is empty or 'raw_code_content' column is missing. Skipping cleaning demonstration."
    )

# %% [markdown]
# ### Save Metadata file

# %%
if not df.empty:
    entries = ["problem_id", "cleaned_code", "tags"]

    existing_columns = [col for col in entries if col in df.columns]

    if "cleaned_code" not in df.columns and "raw_code" in df.columns:
        print(
            "Warning: 'cleaned_code' not found, metadata might not be fully processed for next steps."
        )

    df_metadata = df[existing_columns]

    try:
        df_metadata.to_csv(METADATA_FILE, index=False)
        print(
            f"\nSuccessfully saved metadata for {len(df_metadata)} records to {METADATA_FILE}"
        )
        print("\nSample of metadata.csv content:")
        print(df_metadata.head())
    except Exception as e:
        print(f"Error saving metadata file: {e}")

else:
    print("\nDataFrame is empty. No metadata file created.")
