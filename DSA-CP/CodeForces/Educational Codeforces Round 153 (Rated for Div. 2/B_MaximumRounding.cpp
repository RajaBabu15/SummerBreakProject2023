#include<iostream>
#include<string>
#include<sstream>
#include<cmath>
using namespace std;
long long maximumRounding(string s, int len){
    for(int i=0;i<len;i++){
        if(s[i]-'0'>=5){
            if(i==0){
                long long ret =1;
                return pow(10,len);
            }
            else{
                for(int j=i;j<len;j++){
                    s[j]='0';
                }
                for(int j=i-1;j>=0;j--){
                    if(s[j]-'0'!=9){
                        s[j]+=1;
                        // long long ret_val ;
                        // stringstream ss(s);
                        // ss>>ret_val;
                        // return ret_val;
                        break;
                        
                    }
                }
            }
        }
    }
    long long ret_val ;
    stringstream ss(s);
    ss>>ret_val;
    return ret_val;
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
        cout<<maximumRounding(s,len)<<endl;
    }
    return 0;
}