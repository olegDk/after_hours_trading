def logon() -> dict:
    return {
        'messageId': 'logon',
        'seqNo': None,
        'clientName': '',
        'clientVersion': ''
    }


def logon_response() -> dict:
    return {
        'messageId': 'logon',
        'refNo': None,
        'sessionKey': ''
    }


def keep_alive() -> dict:
    return {
        'messageId': 'keepAlive',
        'sessionKey': ''
    }


def keep_alive_acknowledgement() -> dict:
    return {
        'messageId': 'keepAlive',
        'status': ''
    }


def logon_error_response() -> dict:
    return {
        'messageId': 'logon',
        'refNo': None,
        'error': ''
    }


def subscribe() -> dict:
    return {
        'messageId': 'subscribe',
        'seqNo': None,
        'sessionKey': '',
        'symbols': None
    }


def subscribe_response() -> dict:
    return {
        'messageId': 'marketData',
        'refNo': None,
        'data': []
    }


def market_data() -> dict:
    return {
        'messageId': 'marketData',
        'data': []
    }


def order_request() -> dict:
    return {
        'messageId': 'orderRequest',
        'seqNo': None,
        'sessionKey': '',
        'order': {
            'cid': '',
            'data': {
                'TIF': 'Extended',
                'limit': None,
                'side': '',
                'size': None,
                'symbol': '',
                'username': '',
                'venue': '',
                'target': None
            }
        }
    }


def order_response() -> dict:
    return {
        'messageId': 'orderResponse',
        'refNo': None,
        'order': {
            'cid': '',
            'response': '',
            'data': {
                'symbol': '',
                'username': '',
                'orderId': '',
                'clientId': '',
                'time': None,
                'side': '',
                'limit': None,
                'size': None,
                'venue': '',
                'mnemonic': '',
                'TIF': '',
                'pendingSize': None,
                'BPU': None,
                'CPNL': None
            }
        }
    }


def order_complete() -> dict:
    return {
        'messageId': 'orderStatus',
        'order': {
            'cid': '',
            'id': '',
            'status': ''
        }
    }
