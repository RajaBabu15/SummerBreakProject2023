#include<bits/stdc++.h>

using namespace std;

int main(){
    int t;
    cin>>t;
    while(t--){
        int n,inp;
        cin>>n;
        vector<int> vec(n,0);
        long long ans=1;
        int zeroCounter=0;
        for(int i=0;i<n;i++){
            cin>>inp;
            if(inp!=0) ans*=inp;
            else zeroCounter++;
            vec[i]=inp;
        }
        if(zeroCounter>1) {cout<<0<<endl; }
        else if(zeroCounter==1){cout<<ans<<endl; }
        else {    long long max=ans,val;
        for(int i=0;i<n;i++){
            val=ans/vec[i]*(vec[i]+1);
            if(val>max) max=val;
        }
        cout<<max<<endl;}
    }
    return 0;
}