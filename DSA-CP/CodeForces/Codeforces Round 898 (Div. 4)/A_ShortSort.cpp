#include<bits/stdc++.h>

using namespace std;

int main(){
    int t;
    cin>>t;
    while(t--){
        string s;
        cin>>s;
        if(s=="abc"||s=="acb"||s=="bac"||s=="cba") cout<<"YES"<<endl;
        else cout<<"NO"<<endl;
    }
    return 0;
}