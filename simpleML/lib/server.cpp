#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#define PORT 554

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    // Создаем сокет
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    // Привязываем сокет к порту
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    std::cout << "Server is listening on port " << PORT << "..." << std::endl;

    while (true) {
        if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
            perror("Accept failed");
            continue;
        }

        std::cout << "Client connected!" << std::endl;

        // Открываем файл для чтения
        std::ifstream file("perfect_weights.txt", std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file perfect_weights.txt" << std::endl;
            close(new_socket);
            continue;
        }

        // Отправка содержимого файла
        char buffer[1024];
        while (!file.eof()) {
            file.read(buffer, sizeof(buffer));
            int bytesRead = file.gcount();
            send(new_socket, buffer, bytesRead, 0);
        }
        file.close();
        std::cout << "File sent to client." << std::endl;

        close(new_socket);
    }

    close(server_fd);
    return 0;
}