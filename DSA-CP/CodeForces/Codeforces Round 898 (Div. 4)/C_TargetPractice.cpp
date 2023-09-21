#include <bits/stdc++.h>

using namespace std;

int arr[10][10] = {
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 2, 2, 2, 2, 2, 2, 2, 2, 1},
    {1, 2, 3, 3, 3, 3, 3, 3, 2, 1},
    {1, 2, 3, 4, 4, 4, 4, 3, 2, 1},
    {1, 2, 3, 4, 5, 5, 4, 3, 2, 1},
    {1, 2, 3, 4, 5, 5, 4, 3, 2, 1},
    {1, 2, 3, 4, 4, 4, 4, 3, 2, 1},
    {1, 2, 3, 3, 3, 3, 3, 3, 2, 1},
    {1, 2, 2, 2, 2, 2, 2, 2, 2, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

int main()
{
    int t;
    cin >> t;
    while (t--)
    {
        vector<vector<int>> vec_p;
        for (int i = 0; i < 10; i++)
        {
            string str;
            cin >> str;
            std::replace(str.begin(), str.end(), '.', '0');
            std::replace(str.begin(), str.end(), 'X', '1');

            // Convert the string into a vector
            std::vector<int> vec;
            for (char c : str)
            {
                vec.push_back(c - '0');
            }
            vec_p.push_back(vec);
        }
        int ans=0;
        for(int i=0;i<10;i++){
            for(int j=0;j<10;j++){
                ans+=vec_p[i][j]*arr[i][j];
            }
        }
        cout<<ans<<endl;
    }
    return 0;
}