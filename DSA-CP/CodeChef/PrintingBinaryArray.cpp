#include <iostream>
using namespace std;
#include<vector>

int main() {
	// your code goes here
	int t;
	cin>>t;
	while(t--){
	    int n;
	    cin>>n;
	    int a;
	    for(int i=0;i<n;i++){
	        cin>>a;
	        if(a%2==0) cout<<1<<" ";
	        else cout<<0<<" ";
	    }
	    cout<<endl;
	    
	    
	}
	return 0;
}
