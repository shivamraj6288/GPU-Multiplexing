#include <bits/stdc++.h>
using namespace std;

class server_to_client_msg {
    public:
    int num_blocks;

    server_to_client_msg(){}

};

class client_to_server_msg {
    public:
    int num_blocks;
    int pid;
    client_to_server_msg(){}
};

