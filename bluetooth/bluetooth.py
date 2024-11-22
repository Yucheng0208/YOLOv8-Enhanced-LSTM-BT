import bluetooth
import datetime

def start_bluetooth_server():
    # 創建藍牙服務端 Socket
    server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_socket.bind(("", bluetooth.PORT_ANY))
    server_socket.listen(1)
    
    port = server_socket.getsockname()[1]
    print(f"藍牙服務已啟動，等待連接中... (埠號: {port})")
    
    # 註冊服務（可選）
    bluetooth.advertise_service(
        server_socket,
        "BluetoothServer",
        service_classes=[bluetooth.SERIAL_PORT_CLASS],
        profiles=[bluetooth.SERIAL_PORT_PROFILE],
    )
    
    # 等待連接
    client_socket, client_address = server_socket.accept()
    print(f"來自 {client_address} 的連線已建立")
    
    try:
        while True:
            # 接收訊息（1024 位元組）
            data = client_socket.recv(1024).decode("utf-8")
            if not data:
                print("連線已關閉")
                break
            
            # 記錄接收的時間
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] 收到訊息: {data}")
    
    except Exception as e:
        print(f"發生錯誤: {e}")
    
    finally:
        # 關閉連接
        client_socket.close()
        server_socket.close()
        print("藍牙服務已停止")

if __name__ == "__main__":
    start_bluetooth_server()
