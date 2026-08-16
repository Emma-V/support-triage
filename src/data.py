"""
Data preparation. Runs once at the start, on CPU. After that we barely touch it.

What goes in here:
- Download the Bitext dataset (or read it from disk if it is already there).
- Light text cleaning: collapse double spaces, drop empty rows. Nothing more.
- Find rows that are almost the same sentence. The dataset was generated
  automatically, so many rows are variations of one template. Rows like that
  must end up in the same split, otherwise the model is tested on sentences
  it has already seen.
- Split the data into three parts: train (learn from it), val (check while
  tuning), test (check only at the very end).
- Save a fingerprint of the split: the seed used, how many rows in each part.
  This is what lets the lecturer run the code and get exactly the same split,
  and therefore exactly the same numbers as the report.

Note: no print statements and no plots in this file. Only functions that take
a table and return a table. notebooks/01_data.ipynb is what runs them and
shows the results.
"""
