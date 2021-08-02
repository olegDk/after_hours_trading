from messaging.rabbit_sender import RabbitSender
from config.constants import *

rabbit_sender = RabbitSender(RABBIT_MQ_HOST, RABBIT_MQ_PORT)
message = [{'messageId': 'check'}]
rabbit_sender.send_message(message=message, routing_key=ORDER_RELATED_DATA)
