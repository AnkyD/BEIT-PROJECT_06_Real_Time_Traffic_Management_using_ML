import io
import socket
import struct
from PIL import Image

s = socket.socket()
host = '192.168.43.240' #ip of host
port = 12345
print("waiting")
s.bind((host, port))
print("waiting")
s.listen(5)
connection = s.accept()[0].makefile('rb')
print("waiting")
try:
        image_len = struct.unpack('<L',connection.read(struct.calcsize('<L')))[0]
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream)
        image.show()
        print('Image is verified')
        file_name = str(int(time.time()))
        directory = "sample"
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            np.savez(directory + '/' + file_name + '.npz', train=X, train_labels=y)
        except IOError as e:
            print(e)


finally:
    connection.close()
    s.close()
