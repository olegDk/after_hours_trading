import json
import time
import socket
from typing import Tuple
from config.constants import *
from pika.exceptions import StreamLostError, AMQPConnectionError
from pika.adapters.blocking_connection import BlockingConnection
from pika.adapters.blocking_connection import BlockingChannel
from pika import ConnectionParameters


def new_connection(host: str,
                   port: int):
    return BlockingConnection(
        ConnectionParameters(host=host, port=port)
    )


def reload_connection(host: str,
                      port: int) -> Tuple[BlockingConnection, BlockingChannel]:
    connection = new_connection(host, port)
    channel = connection.channel()
    channel.queue_declare(queue=ORDER_RELATED_DATA)
    channel.queue_declare(queue=MARKET_DATA_TYPE)
    channel.queue_declare(queue=NEWS_TYPE)
    return connection, channel


def connect_rabbit(host: str,
                   port: int) -> Tuple[BlockingConnection, BlockingChannel]:
    while True:
        try:
            print(f"========================================================="
                  f"===================================================")
            print(f"Connection attempt to rabbit...")
            con, ch = reload_connection(host, port)
            print(f"Connected to rabbit successfully")
            print(f"========================================================="
                  f"===================================================")
            return con, ch
        except socket.gaierror:
            print(f"Failed connecting to rabbit, maybe it didn't start yet, "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)
        except ConnectionError:
            print(f"Failed connecting to rabbit, ConnectionError "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)
        except AMQPConnectionError:
            print(f"Failed connecting to rabbit, AMQPConnectionError "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)


class RabbitSender:

    def __init__(self,
                 host: str,
                 port: int):
        self.__host = host
        self.__port = port
        self.__connection, self.__channel = connect_rabbit(self.__host,
                                                           self.__port)

    def send_message(self,
                     message: list,
                     routing_key: str):
        message_bytes = json.dumps(message).encode('UTF-8')
        try:
            self.__channel.basic_publish(
                exchange='',
                routing_key=routing_key,
                body=message_bytes,
            )
        except StreamLostError:
            print('Streaming error')
            reload_connection(self.__host,
                              self.__port)
            self.send_message(message,
                              routing_key=routing_key)
        except AMQPConnectionError:
            print('AMQPConnectionError')
            reload_connection(self.__host,
                              self.__port)
            self.send_message(message,
                              routing_key=routing_key)
