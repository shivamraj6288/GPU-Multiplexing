#include <bits/stdc++.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdlib>
#include <time.h>


#include "utility.h"

#define PORT 8080

int max_blocks = 100000;

int get_rand() {
    return rand()%max_blocks;
}
  
int main(int argc, char const* argv[])
{
    srand(time(0));
    int sock = 0, valread, client_fd;
    struct sockaddr_in serv_addr;
    char* hello = "Hello from client";
    char buffer[1024] = { 0 };
    // printf("waiting1 \n");
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return -1;
    }
    
  
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
  
    // Convert IPv4 and IPv6 addresses from text to binary
    // form
    // printf("waiting2 \n");
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)
        <= 0) {
        printf(
            "\nInvalid address/ Address not supported \n");
        return -1;
    }
    
    // printf("waiting3 \n");
    if ((client_fd
         = connect(sock, (struct sockaddr*)&serv_addr,
                   sizeof(serv_addr)))
        < 0) {
        printf("\nConnection Failed \n");
        return -1;
    }
    client_to_server_msg *send_msg = new client_to_server_msg();
    int req_blocks = get_rand();
    send_msg->pid=(int) getpid();
    send_msg->num_blocks = req_blocks;
    send(sock, send_msg, sizeof(send_msg), 0);
    printf("Client send message: pid: %d, num_blocks: %d\n", send_msg->pid, send_msg-> num_blocks);
    
    server_to_client_msg *recv_msg = new server_to_client_msg();
    while (req_blocks>0){
        valread = read(sock, recv_msg, sizeof(recv_msg));
        printf("client recv msg, num_block: %d\n", recv_msg->num_blocks);
        req_blocks-=recv_msg->num_blocks;
        printf("Pending blocks %d \n", req_blocks);
  
    }
    
    // closing the connected socket
    close(client_fd);
    return 0;
}