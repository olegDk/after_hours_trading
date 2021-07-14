import asyncio
import json
import os
import socket
import emulator.messages as messages  # get rid in future
from analytics.trader import Trader

# TAKION_MIDDLEWARE_IP = os.environ.get('TAKION_IP')
# TAKION_MIDDLEWARE_PORT = int(os.environ.get('TAKION_PORT'))
VERSION_OF_APP = os.environ.get('VERSION_OF_APP', '11.1.1')
PUBLIC_APP_IP = os.environ.get('PUBLIC_APP_IP', '3.211.228.129')
SEND_DELAY = int(os.environ.get('SEND_DELAY', '5000'))
SEND_FREQUENCY = int(os.environ.get('SEND_FREQUENCY', '1000'))
TIMEOUT_GRACE_PERIOD = 300

trader = Trader()


async def send_tcp_message(writer: asyncio.StreamWriter, msg: str):
    writer.write(f'{msg}\n'.encode('utf8'))
    await writer.drain()


async def reply(writer: asyncio.StreamWriter, json_msg: dict):
    msg_type = json_msg['action']
    if msg_type == 'stockL1-update':
        order = trader.process_stock_l1_message(json_msg)
        await send_tcp_message(
            writer,
            json.dumps(order)
        )
        print('Order sent')
    elif msg_type == 'order-complete':
        print('Order placed')


async def handle_message(writer: asyncio.StreamWriter, msg: str):
    try:
        json_msg = json.loads(msg)
        await reply(writer, json_msg)
    except json.JSONDecodeError:
        print('Received a msg with incorrect format')
    except Exception as e:
        print(e)


async def send_handshake(writer: asyncio.StreamWriter):
    handshake = messages.handshake_request()
    await send_tcp_message(writer, json.dumps(handshake))
    print('Handshake sent')


async def handle_server(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter):
    try:
        msg = ''
        while True:
            received_byte = await reader.read(1)
            character = str(received_byte, 'UTF-8')
            if character == '\n':
                print(f'Received: {msg}')
                await handle_message(writer, msg)
                msg = ''
            else:
                msg += character
    finally:
        writer.close()


async def start_interaction():
    host = socket.gethostname()
    port = 1234

    reader, writer = await asyncio.open_connection(
        host, port)

    await send_handshake(writer)
    await handle_server(reader, writer)
