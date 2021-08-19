from config.constants import *

def logon() -> dict:
    return {
        MESSAGE_ID: LOGON_TYPE,
        SEQ: None,
        ACCOUNT_ID_KEY: '',
        CLIENT_VERSION_KEY: ''
    }


def logon_response() -> dict:
    return {
        MESSAGE_ID: LOGON_TYPE,
        REF: None,
        SESSION_KEY: ''
    }


def keep_alive() -> dict:
    return {
        MESSAGE_ID: KEEP_ALIVE,
        SESSION_KEY: ''
    }


def keep_alive_acknowledgement() -> dict:
    return {
        MESSAGE_ID: KEEP_ALIVE,
        STATUS: ''
    }


def logon_error_response() -> dict:
    return {
        MESSAGE_ID: LOGON_TYPE,
        REF: None,
        ERROR: ''
    }


def subscribe() -> dict:
    return {
        MESSAGE_ID: SUBSCRIBE,
        SEQ: None,
        SESSION_KEY: '',
        SYMBOLS: None
    }


def subscribe_response() -> dict:
    return {
        MESSAGE_ID: MARKET_DATA_TYPE,
        REF: None,
        DATA: []
    }


def market_data() -> dict:
    return {
        MESSAGE_ID: MARKET_DATA_TYPE,
        DATA: []
    }


def order_request() -> dict:
    return {
        MESSAGE_ID: ORDER_REQUEST,
        SEQ: None,
        SESSION_KEY: '',
        ORDER: {
            CID: '',
            DATA: {
                TIF: EXTENDED,
                LIMIT: None,
                SIDE: '',
                SIZE: None,
                SYMBOL: '',
                ACCOUNT_ID: '',
                VENUE: None,
                TARGET: None
            }
        }
    }


def order_response() -> dict:
    return {
        MESSAGE_ID: ORDER_RESPONSE,
        REF: None,
        ORDER: {
            CID: '',
            RESPONSE: '',
            DATA: {
                SYMBOL: '',
                ACCOUNT_ID: '',
                ORDER_ID: '',
                CLIENT_ID: '',
                TIME: None,
                SIDE: '',
                LIMIT: None,
                SIZE: None,
                VENUE: '',
                MNEMONIC: '',
                TIF: '',
                PENDING_SIZE: None,
                BPU: None,
                CPNL: None
            }
        }
    }


def order_complete() -> dict:
    return {
        MESSAGE_ID: ORDER_STATUS,
        ORDER: {
            CID: '',
            ORDER_ID: '',
            STATUS: ''
        }
    }
