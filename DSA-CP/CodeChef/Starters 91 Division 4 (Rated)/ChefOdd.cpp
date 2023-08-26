#include <iostream>
#include<bits/stdc++.h>
using namespace std;

int main() {
	// your code goes here
	int t;
	cin>>t;
	while(t--){
	    long long n,k;
	    cin>>n>>k;
	    if(k>floor(n/2))
	        cout<<"NO"<<endl;
	    else if((int(ceil(n/2))-k)%2==0)
	        cout<<"YES"<<endl;
	    else
	        cout<<"NO"<<endl;
	}
	return 0;
}
