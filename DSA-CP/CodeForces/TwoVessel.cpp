#include<bits/stdc++.h>

using namespace std;

int main(){
    int t;
    cin>>t;
    while(t--){
        int a,b,c;
        cin>>a>>b>>c;
        int maxi = max(a,b);
        int mini = min(a,b);
        // int step = (maxi-mini)/2;
        // cout<<step<<endl;
        int counter = 0;
        while(maxi>mini){
            cout<<counter<<maxi<<mini<<endl;
            
            counter++;
            maxi-=c;
            mini-=c;
            if(maxi==mini) break;

        }
        cout<<counter<<endl;
    }
    return 0;
}