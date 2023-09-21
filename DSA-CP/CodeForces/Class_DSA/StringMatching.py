def stringMatching(s:str="abcdabcddef",p:str = "def") -> str :
    s_dash=list(map(ord,s))
    p_dash=list(map(ord,p))
    n=len(s_dash)
    m=len(p_dash)
    for i in range(n-m+1):
        flag=False
        if s_dash[i] == p_dash[0]:
            flag=True
            for j in range(1,m,1):
                if not s_dash[i+j] == p_dash[j]:
                    flag = False
                    break
            if flag :
                return p +" is found in the "+ s +" at index "+ str(i)
    return p +" is not found in the "+ s

def prefix_table(pattern):
    prefix = [0]
    for i in range(1, len(pattern)):
        j = prefix[i - 1]
        while j > 0 and pattern[j] != pattern[i]:
            j = prefix[j - 1]
        if pattern[j] == pattern[i]:
            j += 1
        prefix.append(j)
    return prefix
print(prefix_table("ababaca"))


# def prefix_function(pattern):
#     n = len(pattern)
#     prefix = [0]
#     for i in range(1,n):
#         j = prefix[i - 1]
#         print(j,type(j))
#         while j > 0 and pattern[j] !=pattern[i]:
#             j=pattern[j-1]

#         if pattern[i] == pattern[j]:
#             j+=1
#         prefix.append(j)
#     return prefix

# print(prefix_function("ababaca"))

# import unittest

# class TestStringMatching(unittest.TestCase):
#     def setUp(self):
#         self.func = stringMatching

#     def test_string_matching(self):
#         self.assertEqual(self.func(s="abcdabcddef", p="def"), "def is found in the abcdabcddef at index 8")
#         self.assertEqual(self.func(s="000010001010001", p="101"), "101 is found in the 000010001010001 at index 8")
#         self.assertEqual(self.func(s="abcdabcddef", p="xyz"), "xyz is not found in the abcdabcddef")

# if __name__ == '__main__':
#     unittest.main()

        