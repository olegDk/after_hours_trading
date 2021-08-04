from messaging.rabbit_sender import RabbitSender
from config.constants import *

rabbit_sender = RabbitSender(RABBIT_MQ_HOST, RABBIT_MQ_PORT)
message_order = [{'messageId': 'order'}]
message_md = [{'messageId': 'md'}]
rabbit_sender.send_message(message=message_order, routing_key=ORDER_RELATED_DATA)
rabbit_sender.send_message(message=message_md, routing_key=MARKET_DATA_TYPE)
