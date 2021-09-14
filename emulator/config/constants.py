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
POLICY = 'policy'
INIT_PCT = -1e3
STOCK_TO_TIER_PROPORTION = 'stock_to_tier_proportion'
CONTENT = 'content'
RELEVANCE = 'relevance'
AMC_KEY = 'isYesterdaysAMC'
BP_POS = 'BP_POS'

# Message types
MESSAGE_ID = 'messageId'
ORDER_REQUEST = 'orderRequest'
MARKET_DATA_TYPE = 'marketData'
ORDER_STATUS = 'orderStatus'
ORDER_RESPONSE = 'orderResponse'
ORDER_REPORT = 'orderReport'
SUBSCRIBE = 'subscribe'
KEEP_ALIVE = 'keepAlive'
NEWS_TYPE = 'news'

# Plain constants to use in code
MODEL = 'model'
TARGET = 'target'
DATETIME = "datetime"
PCT_NET = 'pct_net'
ORDERS = 'orders'
POSITIONS = 'positions'
SENT_ORDERS_BY_TICKER = 'sent_orders_by_ticker'

# Constants related to orderRequest only
TIF = 'TIF'
LIMIT = 'limit'
EXTENDED = 53
CID = 'cid'
BUY = 'B'
SELL = 'H'
PREDICTION_KEY = 'prediction'

# Constants related to orderResponse/Report only
MNEMONIC = 'mnemonic'
PENDING_SIZE = 'pendingSize'
BPU = 'BPU'
CPNL = 'CPNL'
ORDER_ID = 'id'
TIME = 'time'
STATUS = 'status'
RESPONSE = 'response'
CLIENT_ID = 'clientId'
EXECUTION = 'execution'

# Order request-response data
ORDER = 'order'
ORDER_RELATED_DATA = 'orderRelatedData'
DATA = 'data'
PRICE = 'price'
SIDE = 'side'
SIZE = 'size'
VENUE = 'venue'
ACCOUNT_ID_KEY = 'accountId'
ORDER_DATA = 'orderData'

# Sector names
APPLICATION_SOFTWARE = 'ApplicationSoftware'
BANKS = 'Banks'
OIL = 'Oil'
RENEWABLE_ENERGY = 'RenewableEnergy'
SEMICONDUCTORS = 'Semiconductors'
CHINA = 'China'

# Trade policies
NEUTRAL = 'neutral'
BULL = 'bull'
BEAR = 'bear'
AGG_BULL = 'agg_bull'
AGG_BEAR = 'agg_bear'
LONG_COEF = 'long_coef'
SHORT_COEF = 'short_coef'
DELTA_COEF = 'delta_coef'
INIT_BP = 2000000
BP_KEY = 'bp'
ACCOUNT_INFORMATION = 'account_information'
BP_USAGE_PCT = 1.0
BP_USAGE_PCT_KEY = 'bp_usage_key'
MAX_ORDERS = 16
BEFORE_8_N_ORDERS = int(MAX_ORDERS/8)
BEFORE_8_30_N_ORDERS = int(MAX_ORDERS/4)
BEFORE_9_N_ORDERS = int(MAX_ORDERS/2)
