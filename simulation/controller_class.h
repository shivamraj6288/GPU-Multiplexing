#include <bits/stdc++.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <future>

#include "utility.h"
using namespace std;

#define PORT 8080


class Controller {
    queue<QueueElements> queue;
    int available_blocks;
    bool scheduling_process_running;
    int server_fd,new_socket, valread;

    public:

    Controller() {
        this->available_blocks=0;
        this->scheduling_process_running=false;
    }

    int allocate_blocks(QueueElements *element) {
        return min(element->num_blocks, this->available_blocks/10);
    }

    void process_queue(){
        // cout <<"scheduling thread running" << endl;
        this->scheduling_process_running=true;
        ServerToClientMsg *send_msg = new ServerToClientMsg();
        while(!this->queue->empty()){
            if(this->available_blocks<=0){
                sleep(2);
            }
            else{
                QueueElements element = this->queue.front();
                this->queue.pop();

                send_msg->num_blocks = allocate_blocks(&element);
                this->available_blocks-=send_msg->num_blocks;
                send(element.socket, send_msg, sizeof(send_msg), 0);
                printf("Server send msg, num_blocks: %d\n", send_msg->num_blocks);
                element.num_blocks -= send_msg->num_blocks;
                if(element.num_blocks!=0) this->queue.push(element);
                else close(element.socket);
            }
        
        }

        this->scheduling_process_running=false;
        // cout <<"scheduling thread exited" << endl;
    }
    void reset_max_available_blocks() {
        while(true){
            sleep(3);
            this->available_blocks=1000;
        }
    }

    void bind_server() {
        struct sockaddr_in address;
        int opt = 1;
        int addrlen = sizeof(address);

        if((server_fd=socket(AF_INET,SOCK_STREAM, 0))==0){
            perror("socket failed");
            exit(EXIT_FAILURE);
        }

        if(setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))){
            perror("setsockopt");
            exit(EXIT_FAILURE);
        }
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(PORT);

        if(bind(server_fd, (struct sockaddr*)&address, sizeof(address))<0) {
            perror("bind failed");
            exit(EXIT_FAILURE);
        }
    }

    void start_listen() {
        if (listen(server_fd, 3) < 0) {
            perror("listen");
            exit(EXIT_FAILURE);
        }
    }

    void start() {
        this->bind_server();
        this->start_listen();
        async(this->reset_max_available_blocks);
        while(true){
            new_socket=accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen);
            if(new_socket<0){
                perror("Error in connection");
                continue;
            }
            ClientToServerMsg *recv_msg = new ClientToServerMsg();
            valread = read(new_socket, recv_msg, sizeof(recv_msg));
            QueueElements new_element(new_socket, recv_msg);
            this->pending_queue.push(new_element);
            if(!this->scheduling_process_running){
                async(this->process_queue);
            }

        }
    }

    void stop() {
        shutdown(server_fd, SHUT_RDWR);
    }
};

