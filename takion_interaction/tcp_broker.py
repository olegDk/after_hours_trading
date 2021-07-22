import asyncio
import json
import socket
import emulator.messages as messages
from analytics.trader import Trader

CLIENT_NAME = 'WBTE'
VERSION_OF_APP = '1.2.3.4'
TIMEOUT_GRACE_PERIOD = 180
RECONNECTION_FREQUENCY = 5
N_ATTEMPTS = 10

trader = Trader()
seq_counter = 0


async def send_future(writer: asyncio.StreamWriter, msg: dict):
    global seq_counter
    msg['seqNo'] = seq_counter
    seq_counter += 1
    writer.write(f'{json.dumps(msg)}\n'.encode('utf8'))
    await writer.drain()


async def send_tcp_message(writer: asyncio.StreamWriter, msg: dict):
    await asyncio.wait_for(send_future(writer, msg),
                           timeout=TIMEOUT_GRACE_PERIOD)


async def reply(writer: asyncio.StreamWriter, json_msg: dict):
    msg_type = json_msg['messageId']
    if msg_type == 'marketData':
        orders = trader.process_l1_message(json_msg)
        if orders:
            for order in orders:
                await send_tcp_message(
                    writer,
                    order
                )
            print(orders)
    elif msg_type == 'orderStatus':
        print('Order placed')


async def handle_message(writer: asyncio.StreamWriter, msg: str):
    try:
        json_msg = json.loads(msg)
        await reply(writer, json_msg)
    except json.JSONDecodeError:
        print('Received a msg with incorrect format')


async def send_logon(writer: asyncio.StreamWriter):
    logon = messages.logon()
    logon['clientName'] = CLIENT_NAME
    logon['clientVersion'] = VERSION_OF_APP
    await send_tcp_message(writer, logon)
    print('Logon sent')


async def send_subscribe(writer: asyncio.StreamWriter):
    subscribe = messages.subscribe()
    tickers = trader.get_subscription_list()
    subscribe['symbols'] = tickers
    await send_tcp_message(writer, subscribe)
    print('Subscribe sent')


async def send_keepalive(writer: asyncio.StreamWriter):
    try:
        while True:
            keep_alive = messages.keep_alive()
            await send_tcp_message(writer, keep_alive)
            print('Keep alive sent')
            await asyncio.sleep(1)
    except ConnectionError as e:
        print(e)
        print(f'Connection lost in keepalive')
        await start_connection()
    except TimeoutError as e:
        print(e)
        print(f'Timeout error in keepalive')
        await start_connection()


async def handle_server(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter):
    try:
        msg = ''
        while True:
            received_byte = await asyncio.wait_for(reader.read(1),
                                                   timeout=TIMEOUT_GRACE_PERIOD)
            character = str(received_byte, 'UTF-8')
            if character == '\n':
                print(f'\n\nReceived: {msg}\n\n')
                await handle_message(writer, msg)
                msg = ''
            else:
                msg += character
    except ConnectionError as e:
        print(e)
        print(f'Connection lost in handle server')
    except TimeoutError as e:
        print(e)
        print(f'Timeout error in handle server')
    finally:
        writer.close()


async def start_connection():
    host = socket.gethostname()
    port = 1234

    reader = None
    writer = None

    for i in range(N_ATTEMPTS):
        try:
            reader, writer = await asyncio.open_connection(
                host, port)
        except ConnectionError:
            print(f'Connection attempt no {i+1} out of {N_ATTEMPTS}'
                  f' failed...\n'
                  f'Trying to reconnect again in 5 seconds...')
            await asyncio.sleep(RECONNECTION_FREQUENCY)

        if reader and writer:
            print(f'Connected successfully')
            break

    await send_logon(writer)
    await send_subscribe(writer)

    await asyncio.gather(send_keepalive(writer), handle_server(reader, writer))


async def start_interaction():
    await start_connection()
