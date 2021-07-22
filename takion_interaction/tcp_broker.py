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
ERROR = 'error'
SESSION_KEY = 'sessionKey'
CLIENT_NAME_KEY = 'clientName'
USER_NAME = 'username'
VERSION_OF_APP_KEY = 'clientVersion'
SEQ = 'seqNo'
LOGON_TYPE = 'logon'
MARKET_DATA_TYPE = 'marketData'
ORDER_STATUS = 'orderStatus'
ORDER_RESPONSE = 'orderResponse'
MESSAGE_ID = 'messageId'
SYMBOLS = 'symbols'
ORDER = 'order'
DATA = 'data'
CID = 'cid'
TID = 'orderId'
STATUS = 'status'

trader = Trader()
seq_counter = 0
session_key = ''


async def send_future(writer: asyncio.StreamWriter, msg: dict):
    global seq_counter
    msg[SEQ] = seq_counter
    seq_counter += 1
    writer.write(f'{json.dumps(msg)}\n\n'.encode('utf8'))
    await writer.drain()


async def send_tcp_message(writer: asyncio.StreamWriter, msg: dict):
    await asyncio.wait_for(send_future(writer, msg),
                           timeout=TIMEOUT_GRACE_PERIOD)


async def reply(writer: asyncio.StreamWriter, json_msg: dict):
    global session_key
    msg_type = json_msg['messageId']
    if msg_type == LOGON_TYPE:
        if 'error' in json_msg.keys():
            print(f'Error on logon: {json_msg[ERROR]}')
        else:
            session_key = json_msg[SESSION_KEY]
            print(f'Received successfull logon response, '
                  f'session_key: {session_key}')
    elif msg_type == MARKET_DATA_TYPE:
        orders = trader.process_l1_message(json_msg)
        if orders:
            for order in orders:
                order[SESSION_KEY] = session_key
                order[ORDER][DATA][USER_NAME] = CLIENT_NAME
                await send_tcp_message(
                    writer,
                    order
                )
            print(orders)
    elif msg_type == ORDER_STATUS:
        client_order_id = json_msg[ORDER][CID]
        takion_order_id = json_msg[ORDER][TID]
        status = json_msg[ORDER][STATUS]
        print(f'Status of order with cid: {client_order_id} '
              f'and tid: {takion_order_id}: {status}')


async def handle_message(writer: asyncio.StreamWriter, msg: str):
    try:
        json_msg = json.loads(msg)
        await reply(writer, json_msg)
    except json.JSONDecodeError:
        print('Received a msg with incorrect format')


async def send_logon(writer: asyncio.StreamWriter):
    logon = messages.logon()
    logon[CLIENT_NAME_KEY] = CLIENT_NAME
    logon[VERSION_OF_APP_KEY] = VERSION_OF_APP
    await send_tcp_message(writer, logon)
    print('Logon sent')


async def send_subscribe(writer: asyncio.StreamWriter):
    while True:
        if session_key:
            subscribe = messages.subscribe()
            tickers = trader.get_subscription_list()
            subscribe[SYMBOLS] = tickers
            subscribe[SESSION_KEY] = session_key
            await send_tcp_message(writer, subscribe)
            print('Subscribe sent')
            break
        else:
            await asyncio.sleep(1)


async def send_keepalive(writer: asyncio.StreamWriter):
    try:
        while True:
            if session_key:
                while True:
                    keep_alive = messages.keep_alive()
                    keep_alive[SESSION_KEY] = session_key
                    await send_tcp_message(writer, keep_alive)
                    print('Keep alive sent')
                    await asyncio.sleep(1)
            else:
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
        prev_character = ''
        while True:
            received_byte = await asyncio.wait_for(reader.read(1),
                                                   timeout=TIMEOUT_GRACE_PERIOD)
            character = str(received_byte, 'UTF-8')
            if character == '\n' and prev_character == '\n':
                print(f'\n\nReceived: {msg}\n\n')
                await handle_message(writer, msg)
                msg = ''
                prev_character = ''
            else:
                msg += character
                prev_character = character
    except ConnectionError as e:
        print(e)
        print(f'Connection lost in handle server')
    except TimeoutError as e:
        print(e)
        print(f'Timeout error in handle server')
    finally:
        writer.close()


async def start_connection():
    global seq_counter, session_key
    host = socket.gethostname()
    port = 1234
    seq_counter = 0
    session_key = ''

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

    await asyncio.gather(handle_server(reader, writer),
                         send_subscribe(writer),
                         send_keepalive(writer))


async def start_interaction():
    await start_connection()
