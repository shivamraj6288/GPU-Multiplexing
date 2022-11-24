rm client
rm server
g++ client.cpp -o client
g++ -std=c++11 -pthread -o server controller.cpp