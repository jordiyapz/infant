import socketio
from time import sleep

if __name__ == '__main__':
    sio = socketio.Client()

    @sio.event
    def connect():
        print("I'm connected!")

    @sio.event
    def connect_error(data):
        print("Connection failed!")

    @sio.event
    def disconnect():
        print("I'm disconnected!")
    
    @sio.event
    def render(data):
        # print(event)
        # print(sid)
        
        print(data)
    

    sio.connect("http://localhost:3000")
    print('my sid is', sio.sid)
    sleep(2)
    sio.disconnect()

    