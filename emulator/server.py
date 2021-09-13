import numpy as np
import asyncio
import traceback
import socket
from config import messages
from config.constants import *
import json
import random
import uuid
from market_data_generator import MarketDataGenerator

logon = False
subscribe = False
mdg = MarketDataGenerator()
md_types = [MARKET_DATA_TYPE, NEWS_TYPE]
md_prob_dist = [0.9, 0.1]


def generate_id() -> str:
    _id = uuid.uuid4().__str__().strip()
    _id = _id.replace('-', '')
    return _id


async def send_to_analytics_server(writer: asyncio.StreamWriter, msg: str):
    writer.write(f'{msg}\n\n'.encode('utf8'))
    await writer.drain()


async def reply(writer: asyncio.StreamWriter, json_msg: dict):
    global logon, subscribe
    msg_type = json_msg[MESSAGE_ID]
    if msg_type == LOGON_TYPE:
        print('Received logon\n\n')
        logon_response = messages.logon_response()
        session_key = generate_id()
        logon_response[SESSION_KEY] = session_key
        logon_response[REF] = json_msg[SEQ]
        logon_response[DATA] = mdg.sample_acc_snapshot()
        await send_to_analytics_server(
            writer,
            json.dumps(logon_response)
        )
        logon = True
    elif msg_type == ORDER_REQUEST:
        order_response = messages.order_response()
        order_response[REF] = json_msg[SEQ]
        await send_to_analytics_server(
            writer,
            json.dumps(order_response)
        )
        sleeping_time = random.randint(1, 10) * 0.1
        await asyncio.sleep(sleeping_time)
        order_report = messages.order_report()
        order_report[DATA][CID] = json_msg[ORDER][CID]
        order_report[DATA][SYMBOL] = json_msg[ORDER][DATA][SYMBOL]
        order_report[DATA][SIZE] = json_msg[ORDER][DATA][SIZE]
        order_report[DATA][SIDE] = json_msg[ORDER][DATA][SIDE]
        await send_to_analytics_server(
            writer,
            json.dumps(order_report)
        )
        sleeping_time = random.randint(1, 10) * 0.1
        await asyncio.sleep(sleeping_time)
    elif msg_type == SUBSCRIBE:
        print('Received subscribe\n\n')
        subscribe_response = messages.subscribe_response()
        subscribe_response[REF] = json_msg[SEQ]
        await send_to_analytics_server(
            writer,
            json.dumps(subscribe_response)
        )
        subscribe = True


async def handle_message(writer: asyncio.StreamWriter, msg: str):
    try:
        json_msg = json.loads(msg)
        await reply(writer, json_msg)
    except json.JSONDecodeError:
        print('Received a msg with incorrect format')
    except Exception as e:
        message = f'Handle message in server error: ' \
                  f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
        print(message)
        print(traceback.format_exc())


async def emulate_market_data(writer: asyncio.StreamWriter):
    while True:
        if logon and subscribe:
            while True:
                md_type = np.random.choice(a=md_types,
                                           size=1,
                                           p=md_prob_dist)
                if md_type == MARKET_DATA_TYPE:
                    msg = json.dumps(mdg.sample_l1_update())
                else:
                    msg = json.dumps(mdg.sample_news_update())
                await send_to_analytics_server(writer, msg)
                print('Sent l1:')
                print(f'{msg}\n\n')
                sleeping_time = random.randint(1, 10) * 0.1
                await asyncio.sleep(sleeping_time)
        else:
            await asyncio.sleep(1)


async def handle_client_messages(reader: asyncio.StreamReader,
                                 writer: asyncio.StreamWriter):
    msg = ''
    prev_character = ''
    while True:
        received_byte = await reader.read(1)
        character = str(received_byte, 'UTF-8')
        if character == '\n' and prev_character == '\n':
            print(f'\n\nReceived: {msg}\n\n')
            await handle_message(writer, msg)
            msg = ''
            prev_character = ''
        else:
            msg += character
            prev_character = character


async def handle_client(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter):
    try:
        await asyncio.gather(handle_client_messages(reader, writer),
                             emulate_market_data(writer))
    except Exception as e:
        print('Handle client in server')
        print(e)
    finally:
        writer.close()


async def main():
    host = socket.gethostname()
    port = 11111
    server = await asyncio.start_server(handle_client, host, port)

    address = server.sockets[0].getsockname()
    print(f'Serving on {address}')

    async with server:
        await server.serve_forever()


asyncio.run(main())
