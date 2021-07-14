import asyncio
import socket
import messages
import json
import random
from market_data_generator import MarketDataGenerator

handshake_received = False
mdg = MarketDataGenerator()


async def send_to_analytics_server(writer: asyncio.StreamWriter, msg: str):
    writer.write(f'{msg}\n'.encode('utf8'))
    await writer.drain()


async def reply(writer: asyncio.StreamWriter, json_msg: dict):
    global handshake_received
    msg_type = json_msg['action']
    if msg_type == 'handshake':
        await send_to_analytics_server(
            writer,
            json.dumps(messages.handshake_response())
        )
        handshake_received = True
    elif msg_type == 'order-request':
        await send_to_analytics_server(
            writer,
            json.dumps(messages.order_response())
        )
        sleeping_time = random.randint(1, 10) * 0.1
        await asyncio.sleep(sleeping_time)


async def handle_message(writer: asyncio.StreamWriter, msg: str):
    try:
        json_msg = json.loads(msg)
        await reply(writer, json_msg)
    except json.JSONDecodeError:
        print('Received a msg with incorrect format')
    except Exception as e:
        print(e)


async def emulate_market_data(writer: asyncio.StreamWriter):
    if handshake_received:
        while True:
            msg = json.dumps(mdg.sample_l1_update())
            await send_to_analytics_server(writer, msg)
            print('Sent l1')
            sleeping_time = random.randint(1, 10) * 0.01
            await asyncio.sleep(sleeping_time)


async def handle_client_messages(reader: asyncio.StreamReader,
                                 writer: asyncio.StreamWriter):
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


async def handle_client(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter):
    try:
        await asyncio.gather(handle_client_messages(reader, writer),
                             emulate_market_data(writer))
    except Exception as e:
        print(e)
    finally:
        writer.close()


async def main():
    host = socket.gethostname()
    port = 1234
    server = await asyncio.start_server(handle_client, host, port)

    address = server.sockets[0].getsockname()
    print(f'Serving on {address}')

    async with server:
        await server.serve_forever()


asyncio.run(main())
