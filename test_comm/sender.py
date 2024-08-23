import socket
def sender_task():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 65432))
        data_to_send = "Hello from sender!"
        print("Sending data:", data_to_send)
        s.sendall(data_to_send.encode())
        
        response = s.recv(1024).decode()
        print("Received response:", response)

if __name__ == "__main__":
    sender_task()