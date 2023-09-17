#include<bits/stdc++.h>

using namespace std;

bool sortbysec(const std::pair<int,int> &a, const std::pair<int,int> &b) {
    return (a.first >= b.first);
}

int main(){
    int t;
    cin>>t;
    while(t--){
        int n;
        cin>>n;
        vector<pair<int,int>> vec(n);
        for(int i=0;i<n;i++) {
            cin>>vec[i].first;
            vec[i].second=i+1;
        }
        sort(vec.begin(), vec.end(), sortbysec);
        for(int i=0;i<n;i++) cout<<vec[i].second<<" ";
        cout<<endl;
    }
    return 0;
}