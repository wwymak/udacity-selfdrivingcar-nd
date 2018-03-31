from socketIO_client import SocketIO, LoggingNamespace

def on_tel_response(*args):
    print('on_aaa_response', args)

socketIO = SocketIO('localhost', 8000, LoggingNamespace)
socketIO.on('telemetry', on_tel_response)

socketIO.wait()