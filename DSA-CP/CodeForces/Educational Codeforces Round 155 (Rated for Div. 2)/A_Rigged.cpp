#include <bits/stdc++.h>

using namespace std;
bool compare(const std::pair<int, int> &a, const std::pair<int, int> &b)
{
    return a.first > b.first;
}
int main()
{
    int t;
    cin >> t;
    while (t--)
    {
        int n, w;
        cin >> n;
        vector<pair<int, int>> participant(n - 1);
        vector<int> endurance(n - 1, 0);
        int s0, e0;
        cin >> s0 >> e0;
        for (int i = 0; i < n - 1; i++)
        {
            cin >> participant[i].first >> participant[i].second;
        }
        sort(participant.begin(), participant.end(), compare);

        int max = e0, i;
        for (i = 0; i < n - 1; i++)
        {
            if (participant[i].first < s0)
                break;

            if (max < participant[i].second)
                max = participant[i].second;
        }
        if (i > n - 1 & max < e0)
        {
            cout << s0 - 1 << endl;
        }
        else if (i > n - 1 & !max < e0)
        {
            cout << -1 << endl;
        }
        else
        {
            if (max > e0 || participant[i - 1].first == s0)
                cout << -1 << endl;
            else
                cout << participant[i].first + 1 << endl;
        }
    }
    return 0;
}