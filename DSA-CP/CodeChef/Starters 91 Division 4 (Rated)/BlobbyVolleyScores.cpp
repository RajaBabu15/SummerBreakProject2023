#include <iostream>
#include<string>
using namespace std;

int main() {
	// your code goes here
	int t;
	cin>>t;
	while(t--){
	    int n;
	    cin>>n;
	    string s;
	    cin>>s;
	    char server='A';
	    int scores[2];
	    for(int i=0;i<2;i++){
	        scores[i]=0;
	    }
	    for(int i=0;i<n;i++){
	        if(s[i]==server)
	            scores[server-'A']++;
	        else{
	            server=s[i];
	        } 
	    }
	    cout<<scores[0]<<" "<<scores[1]<<endl;
	}
	return 0;
}
