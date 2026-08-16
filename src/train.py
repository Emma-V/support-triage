"""
Training. This is the file that teaches the model to sort a customer message
into one of the 27 intents.

What goes in here:
- A function that builds the model: take Qwen3, freeze it, and add two things.
  LoRA, which is a small set of extra weights that do get trained instead of
  training the whole model, and a classification head, which is the final
  layer that returns a probability for each of the 27 options.
- One function that takes a run configuration (which LoRA r, how many epochs,
  which learning rate) and returns the results of that run.

Why a function that takes a configuration instead of code written top to
bottom: you are going to run this many times with different values in order to
compare them. If the settings are a parameter, every run is one row in a
table. If they are written inside the code, every run means editing the code,
and then the runs are not comparable.

This file runs on GPU, and it is the only one that does.
"""
