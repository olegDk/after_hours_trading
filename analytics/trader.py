from analytics.inference_module import InferenceModule

class Trader:
    def __init__(self):
        self.__infer_module = InferenceModule()
        self.__positions = {}
        self.__indicators = {}
        self.__factors = {}
        pass

    def process_message(self, msg: dict) -> dict:
        pass

    def __make_decision(self, msg: dict) -> dict:
        pass

    def __make_order(self, msg: dict) -> dict:
        pass