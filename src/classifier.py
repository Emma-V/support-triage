"""
Using the trained model. This is the door that stage 1 opens to everything
after it.

One function: it takes a list of customer messages as plain text, and returns
for each one:
- intent          which of the 27 intents it belongs to
- confidence      how sure the model is, a number between 0 and 1
- top3            the three most likely intents, for when the model hesitates
- category        one of the 11 categories. Not predicted, just looked up from
                  the intent in a fixed table
- urgency         urgency level. Also looked up from the intent, not learned
- model_version   which version of the model produced this answer

Why this is its own file and not a notebook cell: stage 2 (RAG) and stage 3
(drafting) will need to call this function. You can import a function from a
file, you cannot import a cell from a notebook. Once this file exists, the
later stages do not need to know anything about how the model was trained.
"""
