# -*- coding: utf-8 -*-

import threading
import hashlib
import socket
import base64
import time

class websocketThread(threading.Thread):
    def __init__(self, connection):
        super(websocketThread, self).__init__()
        self.connection = connection

    def run(self):
        print('new websocket client joined!')
        reply = 'i got u, from websocket server.'
        length = len(reply)
        while True:
            time.sleep(5)
            #data = self.connection.recv(1024)
            '''re = parse_data(data)
            print (re)'''
            #time.sleep(5)
            self.connection.send(str(length).encode())

def parseHeaders(msg):
    headers = {}
    msg=msg.decode()
    header, data = msg.split('\r\n\r\n', 1)
    for line in header.split('\r\n')[1:]:
        key, value = line.split(': ', 1)
        headers[key] = value
    headers['data'] = data
    return headers


def generateToken(msg):
    key = msg + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
    key=key.encode()
    ser_key = hashlib.sha1(key).digest()
    return base64.b64encode(ser_key).decode()


if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', 9002))
    sock.listen(5)

    while True:
        connection, address = sock.accept()
        try:
            data = connection.recv(1024)
            headers = parseHeaders(data)
            token = generateToken(headers['Sec-WebSocket-Key'])
            connection.send(("\
            HTTP/1.1 101 Switching Protocols\r\n\
            Upgrade: websocket\r\n\
            Connection: Upgrade\r\n\
            Sec-WebSocket-Accept: {}\r\n\r\n").format(token).encode())
            thread = websocketThread(connection)
            thread.start()
        except socket.timeout:
            print('websocket connection timeout')