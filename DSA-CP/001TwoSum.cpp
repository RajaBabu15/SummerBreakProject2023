#include<iostream>
#include<vector>
using namespace std;
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int size = nums.size();
        int i,j;
        bool found = false; // use a flag to indicate if the target is found
        for(i=0;i<size-1 && !found;i++){ // stop the loop if the flag is true
            for(j=i+1;j<size && !found;j++){
                if(nums[i]+nums[j]==target)
                    found = true; // set the flag to true if the target is found
            }
        }
        vector<int> v = {};
        if (found) { // only push back the indices if the target is found
            v.push_back(i-1); // adjust the indices by -1 because of the loop increment
            v.push_back(j-1);
        }
        return v;
    }
};
