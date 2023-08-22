from math import gcd
from functools import reduce
from itertools import permutations

def find_gcd(numbers):
    return reduce(lambda x, y: gcd(x, y), numbers)

def find_permutations(numbers):
    return list(permutations(numbers))

ls = [8,7,6,5,4,3,2,1]
perm_ls = find_permutations(ls)

for en,perm in enumerate(perm_ls):
    even = [val for i, val in enumerate(perm) if i % 2 != 0]
    odd = [val for i, val in enumerate(perm) if i % 2 == 0]
    if(find_gcd(odd)>find_gcd(even)):
        print(en,":",perm ,even, find_gcd(even), odd, find_gcd(odd))
        break


