import asyncio
import json
import traceback
import config.messages as messages
from analytics.trader import Trader
from config.constants import *

trader = Trader()
seq_counter = 1
session_key = ''
session_host = ''


async def send_future(writer: asyncio.StreamWriter, msg: dict):
    global seq_counter
    msg[SEQ] = seq_counter
    seq_counter += 1
    writer.write(f'{json.dumps(msg)}\n\n'.encode('utf8'))
    print(f'Sent message: {msg}')
    await writer.drain()


async def send_tcp_message(writer: asyncio.StreamWriter, msg: dict):
    await asyncio.wait_for(send_future(writer, msg),
                           timeout=TIMEOUT_GRACE_PERIOD)


async def reply(writer: asyncio.StreamWriter, json_msg: dict):
    if MESSAGE_ID in json_msg:
        msg_type = json_msg[MESSAGE_ID]
        if msg_type == LOGON_TYPE:
            handle_logon(json_msg)
        elif msg_type == MARKET_DATA_TYPE:
            await handle_market_data(writer, json_msg)
        elif msg_type == NEWS_TYPE:
            # handle_news(json_msg)
            print('Received news')
        elif msg_type == ORDER_RESPONSE:
            pass
        elif msg_type == ORDER_REPORT:
            pass
        elif msg_type == SUBSCRIBE:
            pass


async def handle_market_data(writer: asyncio.StreamWriter, msg: dict):
    orders_data = trader.process_md_message(msg)
    market_data = msg[DATA]
    print(orders_data)
    if orders_data:
        for order_data in orders_data:
            order = order_data[ORDER_DATA]
            order[SESSION_KEY] = session_key
            order[ORDER][DATA][ACCOUNT_ID_KEY] = ACCOUNT_ID
            await send_tcp_message(
                writer,
                order
            )
        print(orders_data)
        trader.send_order_log_to_mq(log=orders_data)
    trader.send_market_data_to_mq(log=market_data)


def handle_news(msg: dict):
    trader.process_news(msg)


def handle_logon(msg: dict):
    global session_key
    if ERROR in msg.keys():
        print(f'Error on logon: {msg[ERROR]}')
    elif SESSION_KEY in msg.keys():
        session_key = msg[SESSION_KEY]
        print(f'Received successfull logon response, '
              f'session_key: {session_key}')
        if DATA in msg.keys():
            acc_info = msg[DATA]
            trader.update_account_information(acc_info=acc_info)
    else:
        print(f'Error on logon, full message: {msg}')


async def handle_message(writer: asyncio.StreamWriter, msg: str):
    try:
        json_msg = json.loads(msg)
        await reply(writer, json_msg)
    except json.JSONDecodeError:
        print('Received a msg with incorrect format')


async def send_logon(writer: asyncio.StreamWriter):
    logon = messages.logon()
    logon[ACCOUNT_ID_KEY] = ACCOUNT_ID
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
                    await asyncio.sleep(30)
            else:
                await asyncio.sleep(5)
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
            received_byte =\
                await asyncio.wait_for(reader.read(1),
                                       timeout=TIMEOUT_GRACE_PERIOD)
            character = received_byte.decode('utf-8', 'ignore')
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
    except Exception as e:
        message = f'handle_server error: ' \
                  f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
        print(message)
        print(traceback.format_exc())
        print(f'Is closing: {writer.is_closing()}')
    finally:
        print('Unhandled exception happened, closing connection')
        writer.close()


async def start_connection():
    global seq_counter, session_key, session_host
    port = 11111
    seq_counter = 1
    session_key = ''

    reader = None
    writer = None

    for i in range(N_ATTEMPTS):
        try:
            reader, writer = await asyncio.open_connection(
                session_host, port)
        except ConnectionError as e:
            print(e)
            print(f'Connection attempt no {i+1} out of {N_ATTEMPTS}'
                  f' failed...\n'
                  f'Trying to reconnect again in 5 seconds...')
            await asyncio.sleep(RECONNECTION_FREQUENCY)
            print(f'Trying connection to host {session_host}:{port}')

        if reader and writer:
            print(f'Connected successfully')
            break

    await send_logon(writer)

    await asyncio.gather(handle_server(reader, writer),
                         send_subscribe(writer),
                         send_keepalive(writer))


async def start_interaction(host: str):
    global session_host
    session_host = host
    await start_connection()
