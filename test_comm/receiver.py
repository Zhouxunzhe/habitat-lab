import socket
import threading
import pickle

def handle_sender(conn, addr, data_queue):
    print(f"Connected to {addr}")
    data_length = int.from_bytes(conn.recv(4), 'big')
    data_serialized = conn.recv(data_length)
    data_received = pickle.loads(data_serialized)
    print(f"Data received from {addr}: {data_received}")
    data_queue.append(data_received)
    conn.close()

def receiver_task():
    data_queue = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 65432))
        s.listen()
        print("Main receiver waiting for connections...")
        while len(data_queue) < 2:
            conn, addr = s.accept()
            thread = threading.Thread(target=handle_sender, args=(conn, addr, data_queue))
            thread.start()
            thread.join()

    # Example: Combine data from both senders
    processed_data = {}
    for data in data_queue:
        processed_data.update(data)  # Example operation to merge dictionaries

    # Send processed data to another receiver
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 65434))
        data_serialized = pickle.dumps(processed_data)
        s.sendall(len(data_serialized).to_bytes(4, 'big'))
        s.sendall(data_serialized)
        print("Processed data sent to final receiver")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 65435))
        data_serialized = pickle.dumps(processed_data)
        s.sendall(len(data_serialized).to_bytes(4, 'big'))
        s.sendall(data_serialized)
        print("Processed data sent to final receiver")


if __name__ == "__main__":
    receiver_task()