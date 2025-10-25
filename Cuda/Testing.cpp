#include <iostream>
#include <vector>
using namespace std;

int main()
{
  auto hello = [](int x){
    if(x%2==0)
    {
      return x;
    }


  };
  
  cout << hello(8) << "What is going on here";

}