class Solution {
public:
    int reverse(int x){
        int rev=0;
        while(x!=0){
            rev = rev*10+x%10;
            x/=10;
        }
        return rev;
    }
    bool isPalindrome(int x){
        return x==reverse(x);
    }
};