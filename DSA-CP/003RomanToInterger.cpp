#include<string>
using namespace std;
class Solution {
    
public:
    // - - a b - -
    int nextsignfun(char a,char b){
        if(char_to_int(a)<char_to_int(b)) return -1;
        else return 1;
    }
    int char_to_int(char ch){
        switch(ch){
            case 'I' :return 1; 
            case 'V' :return 5;
            case 'X' :return 10;
            case 'L' :return 50;
            case 'C' :return 100;
            case 'D' :return 500;
            case 'M' :return 1000;
        }
        return 0;
    }
    int romanToInt(string s) {
        int number=0;
        int nextsign=1;
        int len = s.length();
        for (int i=len-1;i>0;i--){
            number+=nextsign*char_to_int(s.at(i));
            nextsign = nextsignfun(s.at(i-1),s.at(i)); 
        }
        number+=nextsign*char_to_int(s.at(0));
        return number;
    }
};