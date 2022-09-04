#include<bits/stdc++.h>
using namespace std;
int a;

void fn(int *a) {
    *a=10;
}

int main () {
    a=20;
    fn(&a);
    cout << a << endl;
}