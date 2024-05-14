# %%
from random import sample
from math import log

import pandas as pd
# %%
class Rational_rules:
  def __init__(self, p_rules, cap=10):
    self.NON_TERMINALS = [x[0] for x in p_rules]
    self.PRODUCTIONS = {}
    self.CAP = cap
    for rule in p_rules:
      self.PRODUCTIONS[rule[0]] = rule[1]

  def generate_tree(self, logging=True, tree_str='S', log_prob=0., depth=0):
    current_nt_indices = [tree_str.find(nt) for nt in self.NON_TERMINALS]
    # Sample a non-terminal for generation
    to_gen_idx = sample([idx for idx, el in enumerate(current_nt_indices) if el > -1], 1)[0]
    to_gen_nt = self.NON_TERMINALS[to_gen_idx]
    # Do generation
    leaf = sample(self.PRODUCTIONS[to_gen_nt], 1)[0]
    to_gen_tree_idx = tree_str.find(to_gen_nt)
    tree_str = tree_str[:to_gen_tree_idx] + leaf + tree_str[(to_gen_tree_idx+1):]
    # Update production log prob
    log_prob += log(1/len(self.PRODUCTIONS[to_gen_nt]))
    # Increase depth count
    depth += 1

    # Recursively rewrite string
    if any (nt in tree_str for nt in self.NON_TERMINALS) and depth <= self.CAP:
      return self.generate_tree(logging, tree_str, log_prob, depth)
    elif any (nt in tree_str for nt in self.NON_TERMINALS):
      if logging:
        print('====DEPTH EXCEEDED!====')
      return None
    else:
      if logging:
        print(tree_str, log_prob)
      return tree_str, log_prob


# %%
# productions = [
#   ['S', ['and(S,S)', 'A']],
#   ['A', ['equa(B)', 'less(B)', 'nequ(B)']],
#   ['B', ['C', 'D']],
#   ['C', ['shape(E),shape(E)', 'color(E),color(E)', 'texture(E),texture(E)']],
#   ['D', ['shape(E),F', 'color(E),G', 'texture(E),H']],
#   ['E', ['a', 'b', 'c']],
#   ['F', ['square', 'circle', 'triangle']],
#   ['G', ['light', 'medium', 'dark']],
#   ['H', ['plain', 'stripe']],

# ]
productions = [
  ['S', ['and(S,S)', 'A']],
  ['A', ['same(B,C)', 'unique(B,C)']],
  ['B', ['everything', 'D', 'E']],
  ['C', ['true', 'color', 'shape', 'texture', 'F']],
  ['D', ['one', 'G']],
  ['E', ['two', 'H']],
  ['F', ['square', 'circle', 'triangle', 'light', 'medium', 'dark', 'plain', 'stripe']],
  ['G', ['a', 'b', 'c']],
  ['H', ['ab', 'ac', 'bc']],

]
test = Rational_rules(productions, cap=100)

#test.generate_tree()

# %% Collect some sample rules
num_iterations = 100000
outputs = []
for _ in range(num_iterations):
    x = test.generate_tree(logging=False)
    if x is not None:
      outputs.append(x)


df = pd.DataFrame(outputs, columns=['program', 'lp'])

grouped_df = df.groupby('program')['lp'].mean().reset_index()
count_df = df['program'].value_counts().reset_index()
count_df.columns = ['program', 'count']
result_df = pd.merge(grouped_df, count_df, on='program')
result_df = result_df.sort_values(by='count', ascending=False).reset_index()


result_df[['program', 'lp', 'count']].to_csv('programs_2.csv', index=False)

# %%
