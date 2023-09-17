import numpy as np

def insertion_sort(arr):
    for i in range(len(arr)):
        temp = arr[i]
        ite=0
        print('{',end='')
        for j in range(i,0,-1):
            ite=j
            print(j,end=',')
            if temp<arr[j]:
                arr[j]=arr[j-1]
        arr[ite]=temp

        print('}',i,arr)
    print(arr)  
x=np.random.randint(low=0,high=10,size=10)
insertion_sort(arr=x)
