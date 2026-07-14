# Phase 1: Load the data and take a first look at it.
#
# Before doing anything else in a data project, you want to answer:
#   - How much data do I have? (shape)
#   - What columns exist, and what are their exact names? (columns)
#   - Is anything missing? (missing values)
#   - What do the numbers roughly look like? (basic stats)
#
# This script does nothing "smart" yet - it's just look-before-you-touch.

import pandas as pd

# Read the CSV into a DataFrame (pandas' table-like data structure).
df = pd.read_csv("data/pgaTourData.csv")

# .shape gives (number of rows, number of columns)
print("Shape (rows, columns):", df.shape)

print("\nColumn names:")
for col in df.columns:
    print(f"  - {col}")

# .isnull().sum() counts, per column, how many rows have a missing value.
print("\nMissing values per column:")
print(df.isnull().sum())

# .describe() gives count/mean/std/min/max/quartiles for numeric columns -
# a quick sanity check on the scale and spread of each stat.
print("\nBasic statistics (numeric columns):")
print(df.describe())

# Show a few actual rows so we can see what real data looks like,
# not just summary numbers.
print("\nFirst 5 rows:")
print(df.head())
