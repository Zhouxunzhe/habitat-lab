from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        
        print("data_load:")
        print(json.dumps(data, indent=2))
        

        action_1 = input("nextaction_agent0:")
        action_2 = input("nextaction_agent1:")
        
        response = {
            "robot_1_action": action_1,
            "robot_2_action": action_2
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=RequestHandler, port=10022):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Start--------port:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()