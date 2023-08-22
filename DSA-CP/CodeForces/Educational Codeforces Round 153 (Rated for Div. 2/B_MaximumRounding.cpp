#include<iostream>
#include<string>
using namespace std;
long long maximumRounding(string s, int len){
    for(int i=0;i<len;i++){
        if(s[i]-'0'>=5){
            if(i==0){
                long long ret =1;
                return ret<<len;
            }
            else{
                for(int j=i;j<len;j++){
                    s[j]='0';
                }
                bool flag_changed=false;
                for(int j=i-1;j>=0;j--){
                    if(s[j]-'0'!=9){
                        s[j]+=1;
                        flag_changed=true;
                    }
                }
                if(!flag_changed){
                    long long ret =1;
                    return ret<<len;
                }
            }
        }
    }
}
int main(){
    ios::sync_with_stdio(false);
    
    int t;
    cin>>t;
    while(t--){
        long long num;
        cin>>num;
        string s = to_string(num);
        int len=s.length();
        cout<<maximumRounding(s,len);
    }
    return 0;
}