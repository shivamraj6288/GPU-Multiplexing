#include <bits/stdc++.h>
using namespace std;

class ServerToClientMsg {
    public:
    int num_blocks;

    ServerToClientMsg(){}

};

class ClientToServerMsg {
    public:
    int num_blocks;
    int pid;
    ClientToServerMsg(){}
};

class QueueElements{
    public:
    int socket;
    int num_blocks;
    int pid;

    QueueElements(int _socket=0, int _num_blocks=100, int _pid=0){
        this->socket=_socket;
        this->num_blocks=_num_blocks;
        this->pid=_pid;
    }
    QueueElements(int _socket, ClientToServerMsg *msg){
        this->socket = _socket;
        this->num_blocks= msg->num_blocks;
        this->pid=msg->pid;
    }


};

