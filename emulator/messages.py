def handshake_response() -> dict:
    return {
        'action':'handshake-response',
        'version': '1.2.3.5',
        'MarketSorterConnected': True,
        'ExecutorConnected': True,
    }


def market_data() -> dict:
    return {
        'action': 'l1',
        'ticker': 'SPY',
        '%BidNet': 0,
        '%AskNet': 0
    }


def order_response() -> dict:
    return {
        'action': 'orderSetSuccessfully',
        'ticker': 'SQ',
        'TIF': 'Extended',
        'ordertype': 'Limit',
        'side': 'Sell',
        'price': 235.90,
        'size': 200,
        'target': 230.00,
        'destination': 'ARCA',
        'listing': 'NYSE'
    }


def order_request() -> dict:
    return {
        'action': 'setOrder',
        'ticker': 'SQ',
        'TIF': 'Extended',
        'ordertype': 'Limit',
        'side': 'Sell',
        'price': 235.90,
        'size': 200,
        'target': 230.00,
        'destination': 'ARCA',
        'listing': 'NYSE'
    }
