# Connection constants
RABBIT_MQ_HOST = 'localhost'
RABBIT_MQ_PORT = 5672
CLIENT_VERSION_KEY = 'clientVersion'
ACCOUNT_ID = 'WBTE'
VERSION_OF_APP = '1.2.3.4'
TIMEOUT_GRACE_PERIOD = 180
RECONNECTION_FREQUENCY = 5
N_ATTEMPTS = 10
ERROR = 'error'
VERSION_OF_APP_KEY = 'clientVersion'
SEQ = 'seqNo'
REF = 'refNo'
LOGON_TYPE = 'logon'
SESSION_KEY = 'sessionKey'
SYMBOLS = 'symbols'

# Stock info constants
SYMBOL = 'symbol'
PCT_BID_NET = 'pctBidNet'
PCT_ASK_NET = 'pctAskNet'
BID = 'bid'
ASK = 'ask'
BID_VENUE = 'bidVenue'
ASK_VENUE = 'askVenue'
CLOSE = 'close'
L1 = 'l1'
INIT_PCT = 0 # change to -1e3 in future

# Message types
MESSAGE_ID = 'messageId'
ORDER_REQUEST = 'orderRequest'
MARKET_DATA_TYPE = 'marketData'
ORDER_STATUS = 'orderStatus'
ORDER_RESPONSE = 'orderResponse'
SUBSCRIBE = 'subscribe'
KEEP_ALIVE = 'keepAlive'

# Plain constants to use in code
MODEL = 'model'
TARGET = 'target'
DATETIME = "datetime"
PCT_NET = 'pct_net'

# Constants related to orderRequest only
TIF = 'TIF'
LIMIT = 'limit'
EXTENDED = 1  # or 1 if integer
CID = 'cid'
BUY = 'B'
SELL = 'S'

# Constants related to orderResponse only
MNEMONIC = 'mnemonic'
PENDING_SIZE = 'pendingSize'
BPU = 'BPU'
CPNL = 'CPNL'
ORDER_ID = 'orderId'
TIME = 'time'
STATUS = 'status'
RESPONSE = 'response'
CLIENT_ID = 'clientId'

# Order request-response data
ORDER = 'order'
ORDER_RELATED_DATA = 'orderRelatedData'
DATA = 'data'
PRICE = 'limit'
SIDE = 'side'
SIZE = 'size'
VENUE = 'venue'
ACCOUNT_ID_KEY = 'accountId'
ORDER_DATA = 'orderData'
