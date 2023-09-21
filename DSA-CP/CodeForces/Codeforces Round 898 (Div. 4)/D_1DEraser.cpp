#include<bits/stdc++.h>

using namespace std;

int main(){
    int t;
    cin>>t;
    while(t--){
        int n,k;
        int cnt=0;
        cin>>n>>k;
        int ptr=0;
        char s;
        for(int i=0;i<n;i++){
            cin>>s;
            if(ptr!=0){ptr--;  continue;}
            if(s=='B'){
                ptr=k-1;
                cnt++;
            }
        }
        cout<<cnt<<endl;
    }
    return 0;
}