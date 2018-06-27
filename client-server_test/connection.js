
const sock = new WebSocket('ws://localhost:9002');

sock.onmessage = function(event){  
    alert(event.data)
    //alert(event.data)
};
sock.onopen=function() {
    alert("opening")
}
sock.onclose=function() {
    alert("closeing")
}
sock.send("test")

