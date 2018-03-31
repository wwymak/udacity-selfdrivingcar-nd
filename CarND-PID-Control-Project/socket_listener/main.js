(function () {
    const socket = io('http://localhost:4567');
    socket.on('telemetry', function (data) {
        console.log(data);
    });

})();

// var io = require('socket.io-client')
// var socket = io('http://localhost:4567', {reconnect: true,  autoConnect: false});
// socket.open();
// console.log('2');
//
//
//
// // var socket = require('socket.io-client')('http://localhost:4567');
// socket.on('connect', function(){console.log('connect')});
// socket.on('event', function(data){console.log(data)});
// socket.on('telemetry', function(data){console.log(data, 'TE')});
// socket.on('disconnect', function(){console.log('disconnect')});
//
// const WebSocket = require('ws');
//
// const ws = new WebSocket('ws://localhost:4567');
//
//
//
// ws.on('open', function open() {
//     console.log('connected');
// });
//
// ws.on('close', function close() {
//     console.log('disconnected');
// });
//
// ws.on('message', function incoming(data) {
//     console.log('jere')
//     console.log(data, 'message');
//
//
// });
//
// ws.on('telemetry', function incoming(data) {
//     console.log(data, 'te');
//
//
// });