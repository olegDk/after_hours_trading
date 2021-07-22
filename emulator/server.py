import asyncio
import socket
import messages
import json
import random
from market_data_generator import MarketDataGenerator

logon = False
subscribe = False
mdg = MarketDataGenerator()


async def send_to_analytics_server(writer: asyncio.StreamWriter, msg: str):
    writer.write(f'{msg}\n'.encode('utf8'))
    await writer.drain()


async def reply(writer: asyncio.StreamWriter, json_msg: dict):
    global logon, subscribe
    msg_type = json_msg['messageId']
    if msg_type == 'logon':
        print('Received logon\n\n')
        await send_to_analytics_server(
            writer,
            json.dumps(messages.logon_response())
        )
        logon = True
    elif msg_type == 'orderRequest':
        await send_to_analytics_server(
            writer,
            json.dumps(messages.order_response())
        )
        sleeping_time = random.randint(1, 10) * 0.1
        await asyncio.sleep(sleeping_time)
    elif msg_type == 'subscribe':
        print('Received subscribe\n\n')
        await send_to_analytics_server(
            writer,
            json.dumps(messages.subscribe_response())
        )
        subscribe = True


async def handle_message(writer: asyncio.StreamWriter, msg: str):
    try:
        json_msg = json.loads(msg)
        await reply(writer, json_msg)
    except json.JSONDecodeError:
        print('Received a msg with incorrect format')
    except Exception as e:
        print("Handle message in server")
        print(e)


async def emulate_market_data(writer: asyncio.StreamWriter):
    await asyncio.sleep(3)
    if logon and subscribe:
        while True:
            msg = json.dumps(mdg.sample_l1_update())
            await send_to_analytics_server(writer, msg)
            print('Sent l1:')
            print(f'{msg}\n\n')
            sleeping_time = random.randint(1, 10) * 0.1
            await asyncio.sleep(sleeping_time)


async def handle_client_messages(reader: asyncio.StreamReader,
                                 writer: asyncio.StreamWriter):
    msg = ''
    while True:
        received_byte = await reader.read(1)
        character = str(received_byte, 'UTF-8')
        if character == '\n':
            print(f'\n\nReceived: {msg}\n\n')
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
        print("Handle client in server")
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
