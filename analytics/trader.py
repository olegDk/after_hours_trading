from analytics.inference_module import InferenceModule
import emulator.messages as messages


class Trader:
    def __init__(self):
        self.__infer_module = InferenceModule()
        self.__stocks_l1 = {}
        self.__positions = {}
        self.__factors = {}

    def process_stock_l1_message(self, msg: dict) -> dict:
        self.__stocks_l1[msg["symbol"]]["%bidNet"] = msg["%bidNet"]
        self.__stocks_l1[msg["symbol"]]["%askNet"] = msg["%askNet"]
        
        order = self.__make_order(msg=msg)
        return order

    def __make_decision(self, msg: dict) -> dict:
        pass

    def __make_order(self, msg: dict) -> dict:
        order = messages.order_request()
        order['symbol'] = msg['symbol']
        return order