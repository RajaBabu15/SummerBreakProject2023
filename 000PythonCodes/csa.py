def csa_levels(summands):
  """Calculates the number of CSA levels needed to reduce a given number of summands to 2.

  Args:
    summands: The number of summands.

  Returns:
    The number of CSA levels needed.
  """

  levels = 0
  while summands > 2:
    summands = (summands + 1) // 2
    levels += 1
  return levels

print(csa_levels(32))
