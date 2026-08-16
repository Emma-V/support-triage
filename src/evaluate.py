"""
Measurement. Once the model is trained, how good is it actually?

What goes in here:
- The scores. The main metric is macro-F1: the score is computed separately
  for each of the 27 intents and then averaged, so a small intent that the
  model fails on completely is not hidden by the big ones.
- A classification report, a table with a score per intent.
- A confusion matrix: which intents the model mixes up with each other. For
  example get_invoice and check_invoice, where the difference is one verb.
- Error slices: check whether the model fails more on rows with typos, or on
  rows that are keywords rather than a full sentence. The dataset marks this
  in the flags column.

The plots and tables that go into the report come from this file.
"""
