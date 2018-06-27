# coding: utf-8
import socket
import struct
import hashlib,base64
import threading
import time
import random
import cv2
from libdetection import detection,drawAttention

'''
+-+-+-+-+-------+-+-------------+-------------------------------+
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-------+-+-------------+-------------------------------+
|F|R|R|R| opcode|M| Payload len |    Extended payload length    |
|I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
|N|V|V|V|       |S|             |   (if payload len==126/127)   |
| |1|2|3|       |K|             |                               |
+-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
|     Extended payload length continued, if payload len == 127  |
+ - - - - - - - - - - - - - - - +-------------------------------+
|                               | Masking-key, if MASK set to 1 |
+-------------------------------+-------------------------------+
| Masking-key (continued)       |          Payload Data         |
+-------------------------------- - - - - - - - - - - - - - - - +
:                     Payload Data continued ...                :
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
|                     Payload Data continued ...                |
+---------------------------------------------------------------+
'''


connectionlist = {} #存放链接客户fd,元组
g_code_length = 0
g_header_length = 0 #websocket数据头部长度
PRINT_FLAG = True

# 计算web端提交的数据长度并返回
def get_datalength(msg):
    global g_code_length
    global g_header_length
    g_code_length = msg[1] & 127
    if g_code_length == 126:
        g_code_length = struct.unpack('>H', msg[2:4])[0]
        g_header_length = 8
    elif g_code_length == 127:
        g_code_length = struct.unpack('>Q', msg[2:10])[0]
        g_header_length = 14
    else:  
        g_header_length = 6  
    g_code_length = int(g_code_length)
    return g_code_length  

# 解析web端提交的bytes信息，返回str信息（可以解析中文信息）
def parse_data(msg):
    global g_code_length
    g_code_length = msg[1] & 127
    if g_code_length == 126:
        g_code_length = struct.unpack('>H', msg[2:4])[0]
        masks = msg[4:8]
        data = msg[8:]
    elif g_code_length == 127:
        g_code_length = struct.unpack('>Q', msg[2:10])[0]
        masks = msg[10:14]
        data = msg[14:]
    else:
        masks = msg[2:6]
        data = msg[6:]
    en_bytes = b""
    cn_bytes = []
    for i, d in enumerate(data):
        nv = chr(d ^ masks[i%4])
        nv_bytes = nv.encode()
        nv_len = len(nv_bytes)
        if nv_len == 1:
            en_bytes += nv_bytes
        else:
            en_bytes += b'%s'
            cn_bytes.append(ord(nv_bytes.decode()))
    if len(cn_bytes) > 2:
        cn_str = ""
        clen = len(cn_bytes)
        count = int(clen / 3)
        for x in range(count):
            i = x * 3
            b = bytes([cn_bytes[i], cn_bytes[i + 1], cn_bytes[i + 2]])
            cn_str += b.decode()
        new = en_bytes.replace(b'%s%s%s', b'%s')
        new = new.decode()
        res = (new % tuple(list(cn_str)))
    else:
        res = en_bytes.decode()
    return res

# 调用socket的send方法发送str信息给web端
def sendMessage(msg):
    global connectionlist
    send_msg = b""  #使用bytes格式,避免后面拼接的时候出现异常
    send_msg += b"\x81"
    back_str = []
    back_str.append('\x81')
    data_length = len(msg.encode()) #可能有中文内容传入，因此计算长度的时候需要转为bytes信息
    if PRINT_FLAG:
        print("INFO: send message is %s and len is %d" % (msg, len(msg.encode('utf-8'))))
    # 数据长度的三种情况
    if data_length <= 125:#当消息内容长度小于等于125时，数据帧的第二个字节0xxxxxxx 低7位直接标示消息内容的长度
        send_msg += str.encode(chr(data_length))
    elif data_length <= 65535:#当消息内容长度需要两个字节来表示时,此字节低7位取值为126,由后两个字节标示信息内容的长度
        send_msg += struct.pack('b', 126)
        send_msg += struct.pack('>h', data_length)
    elif data_length <= (2^64-1):#当消息内容长度需要把个字节来表示时,此字节低7位取值为127,由后8个字节标示信息内容的长度  
        send_msg += struct.pack('b', 127)
        send_msg += struct.pack('>q', data_length)
    else:
        print (u'太长了')
    send_message = send_msg + msg.encode('utf-8')

    for connection in connectionlist.values():
        if send_message != None and len(send_message) > 0:
            connection.send(send_message)

#删除连接,从集合中删除连接对象item
def deleteconnection(item):
    global connectionlist
    del connectionlist['connection'+item]

#定义WebSocket对象(基于线程对象)
class WebSocket(threading.Thread):
    def __init__(self,conn,index,name,remote, path=""):
        #初始化线程
        threading.Thread.__init__(self)
        #初始化数据,全部存储到自己的数据结构中self
        self.conn = conn
        self.index = index
        self.name = name
        self.remote = remote
        self.path = path
        self.GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        self.buffer = ""
        self.buffer_utf8 = b""
        self.length_buffer = 0

    def generate_token(self, WebSocketKey):
        WebSocketKey = WebSocketKey + self.GUID
        Ser_WebSocketKey = hashlib.sha1(WebSocketKey.encode(encoding='utf-8')).digest()
        WebSocketToken = base64.b64encode(Ser_WebSocketKey) # 返回的是一个bytes对象
        return WebSocketToken.decode('utf-8')
    
    #运行线程
    def run(self):
        #Log输出,套接字index启动
        if PRINT_FLAG:
            print('Socket %s Start!' % self.index)
        global g_code_length
        global g_header_length
        self.handshaken = False #Socket是否握手的标志,初始化为false
        while True:
            if self.handshaken == False: #如果没有进行握手
                if PRINT_FLAG:
                    print('INFO: Socket {} Start Handshaken with {}!'.format(self.index,self.remote))
                self.buffer = self.conn.recv(1024).decode('utf-8') #socket会话收到的只能是utf-8编码的信息，将接收到的bytes数据，通过utf-8编码方式解码为unicode编码进行处理
                if PRINT_FLAG:
                    print("INFO: Socket {} self.buffer is {}".format(self.index, self.buffer))
                if self.buffer.find('\r\n\r\n') != -1:
                    headers = {}
                    header, data = self.buffer.split('\r\n\r\n', 1) #按照这种标志分割一次,结果为：header data
                    #对header进行分割后，取出后面的n-1个部分
                    for line in header.split("\r\n")[1:]: #再对header 和 data部分进行单独的解析
                        key, value = line.split(": ", 1) #逐行的解析Request Header信息(Key,Value)
                        headers[key] = value
                    try:
                        WebSocketKey = headers["Sec-WebSocket-Key"]
                    except KeyError:
                        print("Socket {} Handshaken Failed!".format(self.index))
                        deleteconnection(str(self.index))
                        self.conn.close()
                        break
                    WebSocketToken = self.generate_token(WebSocketKey)
                    headers["Location"] = ("ws://{}{}".format(headers["Host"], self.path))
                    #握手过程,服务器构建握手的信息,进行验证和匹配
                    #Upgrade: WebSocket 表示为一个特殊的http请求,请求目的为从http协议升级到websocket协议
                    handshake = "HTTP/1.1 101 Switching Protocols\r\n"\
                            "Connection: Upgrade\r\n"\
                            "Sec-WebSocket-Accept: " + WebSocketToken + "\r\n"\
                            "Upgrade: websocket\r\n\r\n"
                    self.conn.send(handshake.encode(encoding='utf-8')) # 前文以bytes类型接收，此处以bytes类型进行发送
                    # 此处需要增加代码判断是否成功建立连接
                    self.handshaken = True #socket连接成功建立之后修改握手标志
                    #向全部连接客户端集合发送消息,(环境套接字x的到来)
                    sendMessage("Welocomg " + self.name + " !")
                    g_code_length = 0
                else:
                    print("Socket {} Error2!".format(self.index))
                    deleteconnection(str(self.index))
                    self.conn.close()
                    break
            else:
                # 每次接收128字节数据，需要判断是否接收完所有数据，如没有接收完，需要循环接收完再处理
                

                cap = cv2.VideoCapture(0)
                while(1):
                    _, frame = cap.read()
                    frame = detection(frame)
                    cv2.imshow("capture", frame)
                    r=random.random()
                    if r>0.6:
                        sendMessage("1")
                    elif r<0.3:
                        sendMessage("-1")
                    else:
                        sendMessage("0")
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
                
                '''mm = self.conn.recv(128)
                #计算接受的长度，判断是否接收完，如未接受完需要继续接收
                if g_code_length == 0:
                    get_datalength(mm) # 调用此函数可以计算并修改全局变量g_code_length和g_header_length的值
                self.length_buffer += len(mm)
                self.buffer_utf8 += mm
                if self.length_buffer - g_header_length < g_code_length:
                    if PRINT_FLAG:
                        print("INFO: 数据未接收完,接续接受")
                    continue
                else:
                    if PRINT_FLAG:
                        print("g_code_length:", g_code_length)
                        print("INFO Line 204: Recv信息 %s,长度为 %d:" % (self.buffer_utf8, len(self.buffer_utf8)))
                    if not self.buffer_utf8:
                        continue
                    recv_message = parse_data(self.buffer_utf8)
                    if recv_message == "quit":
                        print("Socket %s Logout!" % (self.index))
                        nowTime = time.strftime('%H:%M:%S',time.localtime(time.time()))
                        sendMessage("%s %s say: %s" % (nowTime, self.remote, self.name+" Logout"))
                        deleteconnection(str(self.index))
                        self.conn.close()
                        break
                    else:
                        nowTime = time.strftime('%H:%M:%S',time.localtime(time.time()))
                        sendMessage("%s %s say: %s" % (nowTime, self.remote, recv_message))
                    g_code_length = 0
                    self.length_buffer = 0
                    self.buffer_utf8 = b""'''

#WebSocket服务器对象()
class WebSocketServer(object):
    def __init__(self):
        self.socket = None
        self.i = 0
    #开启操作
    def begin(self):
        if PRINT_FLAG:
            print('WebSocketServer Start!')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ip = 'localhost'
        port = 9002
        if PRINT_FLAG:
            print("WebServer is listening %s,%d" % (ip,port))
        self.socket.bind((ip,port))
        self.socket.listen(50)
        #全局连接集合
        global connectionlist
        
        while True:
            #服务器响应请求,返回连接客户的信息(连接fd,客户地址)
            connection, address = self.socket.accept()
            #根据连接的客户信息,创建WebSocket对象(本质为一个线程)
            #sockfd，index，用户名，地址
            newSocket = WebSocket(connection,self.i,address[0],address)
            #线程启动
            newSocket.start()
            #更新连接的集合(hash表的对应关系)-name->sockfd
            connectionlist['connection'+str(self.i)]=connection
            self.i += 1
            
if __name__ == "__main__":
    server = WebSocketServer()
    server.begin()
