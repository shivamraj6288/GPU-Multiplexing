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

queue<pair<int,int>> pending_queue;
int max_blocks_available=1000;

void process_queue(queue<pair<int,int>> *queue, int *available_blocks){
    server_to_client_msg *send_msg = new server_to_client_msg();
    while(!queue->empty()){
        if(*available_blocks<=0){
            sleep(2);
            continue;
        }
        pair<int,int> pr = queue.top();
        queue.pop();

        send_msg->num_block = min(pr->second, available_blocks/10);
        available_blocks-=blocks;
        send(pr->first, send_msg, sizeof(send_msg), 0);
    }
}




int main (int argc, char* argv[]) {
    int server_fd,new_socket, valread;

    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};
    char *hello = "Hello from server";

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

    printf("waiting1 \n");

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    printf("waiting2 \n");

    set<int> new_sockets;
    while (true) {
        new_socket=accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen);
        if(new_socket<0) {
            perror("error in connection \n");
            continue;
        }
        new_sockets.insert(new_socket);
        printf("Socket Number: %d \n", new_socket);

        client_to_server_msg *recv_msg = new client_to_server_msg();
        server_to_client_msg *send_msg = new server_to_client_msg();
        send_msg->num_blocks=10;
        
        valread = read(new_socket, recv_msg, 1024);
        printf("Server recv msg, pid: %d, num_blocks: %d\n", recv_msg->pid, recv_msg->num_blocks);
        send(new_socket, send_msg, sizeof(send_msg),0);
        printf("Server send msg, num_blocks: %d\n", send_msg->num_blocks);

    }
    // if((new_socket=accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen))<0){
    //     perror("accept");
    //     exit(EXIT_FAILURE);
    // }

    // printf("waiting3 \n");

    // valread = read(new_socket, buffer, 1024);
    // printf("%s\n", buffer);
    // send(new_socket, hello, strlen(hello),0);
    // printf("Hello message sent\n");

    close(new_socket);
    shutdown(server_fd, SHUT_RDWR);
    return 0;
}