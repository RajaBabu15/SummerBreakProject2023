#include <vector>
#include <bits/stdc++.h>

using namespace std;

void mergeTwoSortedArraysWithoutExtraSpace(vector<long long> &a, vector<long long> &b)
{
    // Write your code here.

    /*
    Brute Force Approach
    */
    // vector<long long >ans(a.size()+b.size());
    // int n=a.size(),b.size();
    // int i=0,j=0,counter=0;
    // while(i<n&&j<m){
    // 	if(a[i]<b[j]) ans[counter++]=a[i++];
    // 	else ans[counter++]=b[j++];
    // }
    // while(i<n) ans[counter++]=a[i++];
    // while (j<m) ans[counter++]=b[j++];
    // for(int i=0;i<n;i++) a[i]=ans[i];
    // for(int i=0;i<m;i++) b[i]=ans[i+n];

    /* 1st Optimal Approach*/
    // int n=a.size(),m=b.size();
    // int i=n-1,j=0;
    // while(i>=0&&j<m){
    // 	if(a[i]>b[j]) swap(a[i--],b[j++]);
    // 	else break;
    // }
    // sort(a.begin(),a.end());
    // sort(b.begin(),b.end());

    /* 2ND Optimal Approach == Gap Method */
    /* 2ND Optimal Approach == Gap Method */
    int n = a.size(), m = b.size();
    int size = (n + m);
    int gap = size / 2 + size % 2; // This gives the ceil of the size divided by Two
    while (gap > 0)
    {
        int i = 0, j = gap;
        while (j < n + m)
        {
            if (i < n && j < n)
            {
                if (a[i] > a[j])
                    swap(a[i], a[j]);
            }
            else if (i < n && j >= n)
            {
                if (a[i] > b[j - n])
                    swap(a[i], b[j - n]);
            }
            else
            {
                if (b[i - n] > b[j - n])
                    swap(b[i - n], b[j - n]);
            }
            i++;
            j++;
        }
        gap = gap / 2 + gap % 2;
    }
}

int main()
{
    vector<long long> a = {1, 8, 8};
    vector<long long> b = {2, 3, 4, 5};
    mergeTwoSortedArraysWithoutExtraSpace(a, b);
    for (int i = 0; i < a.size(); i++)
        cout << a[i] << " ";
    cout << endl;
    for (int i = 0; i < b.size(); i++)
        cout << b[i] << " ";
    cout << endl;
    return 0;
}
