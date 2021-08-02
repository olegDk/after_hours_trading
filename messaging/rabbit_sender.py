import json
from typing import Tuple
from config.constants import *
from pika.exceptions import StreamLostError
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
    return connection, channel


class RabbitSender:

    def __init__(self,
                 host: str,
                 port: int):
        self.__host = host
        self.__port = port
        self.__connection, self.__channel = reload_connection(self.__host,
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
            print('streaming error')
            reload_connection(self.__host,
                              self.__port)
            self.send_message(message,
                              routing_key=routing_key)
