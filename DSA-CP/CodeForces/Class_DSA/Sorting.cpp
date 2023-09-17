#include<bits/stdc++.h>

using namespace std;


int main(){
    int arr[]={9,8,7,6,5,4,3,2,10};
    for(int i=1;i<9;i++){
        int temp=arr[i];int j;
        for(j=i-1;j>=0;j--){
            if(arr[j]>temp){
                arr[j+1]=arr[j];
            }
            else {
                break;
            }
        }
        arr[j+1]=temp;
    }
    for(int i=0;i<9;i++) cout<<arr[i]<<" ";
    return 0;
}