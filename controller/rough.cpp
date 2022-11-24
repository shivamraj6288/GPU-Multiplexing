#include<bits/stdc++.h>
#include <thread>
using namespace std;
int a;

void fn() {
    a=10;
}

int main () {
    a=20;
    cout << a << endl;
    thread t1(fn);
    t1.join();
    cout << a << endl;
    
}