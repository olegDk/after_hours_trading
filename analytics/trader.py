from analytics.inference_module import InferenceModule
import emulator.messages as messages

class Trader:
    def __init__(self):
        self.__infer_module = InferenceModule()
        self.__positions = {}
        self.__indicators = {}
        self.__factors = {}

    def process_stockl1_message(self, msg: dict) -> dict:
        order = messages.order_request()
        order['symbol'] = msg['symbol']
        return order

    def __make_decision(self, msg: dict) -> dict:
        pass

    def __make_order(self, msg: dict) -> dict:
        pass