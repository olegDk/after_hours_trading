def handshake_request() -> dict:
    return {
        'action': 'handshake', 
        'version': '1.0.0.1', 
        'ip': '3.211.228.129',
        'sendDelay': 5000, 
        'sendFrequency': 1000}


def handshake_response() -> dict:
    return {
        'action': 'handshake-response',
        'version': '1.2.3.5',
        'marketSorterConnected': True,
        'executorConnected': True,
    }


def market_data() -> dict:
    return {
        'action': 'stockL1-update',
        'symbol': 'Symbol',
        'trd': 'T',
        'pctBidNet': 0,
        'pctAskNet': 0,
        'bidL1': 0,
        'askL1': 0,
        'last': 0,
        'closePrice': 0,
        'premarketVolume': 0
    }

def subscribe_request() -> dict:
    return {
        'action': 'stockL1',
        'username': 'TDXX',
        'symbols': ['MSFT', 'AAPL']
    }


def subscribe_response() -> dict:
    return {
        'action': 'stockL1-response'
    }


def order_request() -> dict:
    return {
        'action': 'order-request',
        'TIF': 'Extended',
        'id': 'id',
        'limit': 0,
        'side': 'B',
        'size': 100,
        'symbol': 'Symbol',
        'username': 'GCXX',
        'venue': 'NSDQ'
    }


def empty_market_data_response() -> dict:
    return {
        'action': 'gotIt'
    }


def order_complete() -> dict:
    return {
        'action': 'order-complete',
        'username': 'GCXX',
        'id': 'id'
    }


def order_response() -> dict:
    return {
        'action': 'orderSetSuccessfully',
        'symbol': 'symbol',
        'TIF': 'Extended',
        'ordertype': 'Limit',
        'side': 'side',
        'price': 235.90,
        'size': 200,
        'target': 230.00,
        'destination': 'ARCA',
        'listing': 'NYSE'
    }
