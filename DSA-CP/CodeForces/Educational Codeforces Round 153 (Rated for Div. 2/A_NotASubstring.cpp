#include<iostream>
#include<stack>
using namespace std;

bool IsRegularBracketSequences(string s, int n){
    bool flag = true;
    int charCounter=0;
    for(int i=0;i<n;i++){
        if(s[i]=='(')
            charCounter++;
        else if(s[i]==')')
            charCounter--;
        if(charCounter<0)
            return false;
    }
    if(charCounter!=0)
        flag=false;
    return flag;
}

void PrintRegularBracketSequences(string s, int n){
    int charCounter=0;
    for(int i=0;i<n;i++){
        if(s[i]=='('){
            cout<<s[i];
            charCounter++;
        }
        else if(s[i]==')'){
            charCounter--;
        }
        if(charCounter<0){
            cout<<'()';
        else
            cout<<')';
    }
}

int main(){
    int t;
    cin>>t;
    while(t--){
        string s;
        cin>>s;
        bool flag = IsRegularBracketSequences(s,s.length());
        if(flag){
            cout<<"NO"<<endl;
        }
        else{
            cout<<"YES"<<endl;
            PrintRegularBracketSequences(s,s.length());
        }

    }
    return 0;
}