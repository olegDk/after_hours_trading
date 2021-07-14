# Official documentation of Takion's middleware
In this project there is a list of the different flows/protocols used in the Takion Middleware. The purpose is to keep track of any changes in the API so that team members are aligned on the data flow.

-------

## Message Flow
In this section the different messages between both ends are described.

**Considerations**:
 - All messages in the connection must be encoded as [UTF-8](https://www.utf8-chartable.de/).
 - All messages must begin with a JSON and end with special the character `/n`.
 - Example message: `{"example":"ok"}\n`.

### Messages
 - [Authentication](#authentication)
 - [Account Request](#account-request)
 - [Account Snapshot](#account-snapshot)
 - [Account Update](#account-update)
 - [Handshake](#handshake)
 - [Handshake Response](#handshake-response)
 - [Hartbeat](#handshake)
 - [Login](#login)
 - [Logout](#logout)
 - [List of Symbols](#list-of-symbols)
 - [List of Symbols Request](#list-of-symbols-request)
 - [Position Snapshot](#position-snapshot)
 - [Position Update](#position-update)
 - [Stock Reload](#stock-reload)
 - [Stock Subscription](#stock-subscription)
 - [Stock Snapshot](#stock-snapshot)
 - [Stock Update](#stock-update)
 - [Unsubscribe to Stock](#unsubscribe-to-stock)


### Handshake
 - *Description:* First request that must be sent when initializing the connection.
 - *Initiatior:* Mobile Server.
 - *Structure:*
```
{
  "action":"handshake",
  "version": (String, version of the backend),
  "ip": (String, ip of the mobile server),
  "sendDelay": (Integer, Optional, number of milliseconds in which ALL the pending stocks updates will be sent),
  "sendFrequency": (Integer, Optional, how often update messages for a portion of the pending stocks will be sent)
}
```
 - *Example:*
```
{
  "action":"handshake",
  "version":"1.0.0.1",
  "ip":"3.211.228.129",
  "sendDelay": 5000,
  "sendFrequency": 1000
}
```
 - Expected response: see [Handshake Response](#handshake-response)
 - Expected response: see [List of Symbols](#list-of-symbols)

### Handshake Response
 - *Description:* Responds to a [Handshake](#handshake). The mobile server can only send subsequent messages if it receives this message (after sending the [handshake](#handshake)).
 - *Initiatior:* Middleware.
 - *Structure:*
```
{
  "action":"handshake-response",
  "version": (String, version of the backend),
  "MarketSorterConnected": (Bool, true only if Market Sorter is connected),
  "ExecutorConnected": (Bool, true only if Executor is connected)
}
```
 - *Example:*
```
{
  "action":"handshake-response",
  "version":"1.0.5.56",
  "MarketSorterConnected":true,
  "ExecutorConnected":false
}
```

 ### Hartbeat
 - *Description:* Message that must be sent at least every 2 minutes to let the other service know the connection should be kept alive.
 - *Initiatior:* Mobile Server / Middleware.
 - *Structure:*
```
{
  "action":"hartbeat"
}
```
 - *Example:*
```
{
  "action":"hartbeat"
}
```

### Login
 - *Description:* Validates a username and password are correct.
 - *Initiatior:* Mobile Server.
 - *Structure:*
```
{
  "action":"login",
  "username": (Username of the user),
  "password": (Password of the user),
  "id": (Integer, Random, unique id of request)
}
```
 - *Example:*
```
{
  "action":"login",
  "username":"my_username",
  "password":"my_password",
  "id": 833733,

}
```
 - Expected response: see [Authentication](#authentication)

### Logout
 - *Description:* Logs a user out, all [subscriptions](#stock-subscription) will be canceled
 - *Initiatior:* Mobile Server.
 - *Structure:*
```
{
  "action":"logout",
  "username": (String, Username of the user)
}
```
 - *Example:*
```
{
  "action":"logout",
  "username":"TXNP"
}
```


### Authentication
 - *Description:* Responds to a [Login](#login).
 - *Initiatior:* Middleware.
 - *Structure:*
```
{
  "action":"authentication",
  "username": (String, username of the user),
  "result": (Bool, true only if the combination username/password is correct),
  "id": (Integer, same as sent in the `login` request)
}
```
 - *Example:*
```
{
  "action":"authentication",
  "username":"my_username",
  "result":true,
  "id": 833733
}
```

### Account Request
 - *Description:* Message sent after a *succesful* [Login](#login), after the [Authentication](#authentication) message. This message requests the [Account Snapshot](#account-snapshot), several [Position Snapshot](#position-snapshot) with the corresponding [Position Update](#position-update), and several [Order Snapshot](#order-snapshot) with the corresponding [Order Update](#order-update).
 - *Initiatior:* Mobile Server.
 - *Structure:*
```
{
  “action”:“account-request”,
  “id”: (Integer, Random, unique id of request)
  “username”:(String, Username of the user)
}
```
 - *Example:*
```
{
  “action”:“account-request”,
  “id”:“00239230903923039203",
  “username”:“TMBL”
}
```

### Account Snapshot
 - *Description:* Message sent after an [Account Request](#account-request).
 - *Initiatior:* Middleware.
 - *Structure:*
```
{
  "action":"account-snapshot",
  "accountId": (String, id of account in Takion system),
  "BP": (Number, buying power),
  "BPU": (Number, buying power used),
  "CPNL": (Number, closed PNL),
  "EBE": (Number, estimated beginning equity),
  "id": (String, id of the message),
  "investment": (Number, Account Balance),
  "OPNL": (Number, Open PNL),
  "StockPosCnt": (Number, how many Stock Positions the Account has),
  "positions": (Array, opened positions. It's length equals StockPosCnt. Absent if StockPosCnt === 0),
  "username":"TMBL"
}
```
 - *Example:*
```
{
  "action":"account-snapshot",
  "id":"ZXCV",
  "accountId":"00000000",
  "BP":1.00,
  "BPU":0.00,
  "investment":0.00,
  "CPNL":0.00,
  "EBE":"102.32",
  "OPNL":0.00,
  "StockPosCnt":0,
  "username":"TMBL"
}
```

### Account Update
 - *Description:* Message sent multiple times after the [Account Snapshot](#account-snapshot) message.
 - *Initiatior:* Middleware.
 - *Structure:*
```
{
  "action":"account-update",
  "username":"TMBL",
  "BP": (Number, Optional, buying power),
  "BPU": (Number, Optional, buying power used),
  "CPNL": (Number, Optional, closed PNL),
  "OPNL": (Number, Optional, Open PNL),
  "investment": (Number, Optional, Account Balance),
  "StockPosCnt": (Number, Optional, how many Stock Positions the Account has),
  "OrderCnt": (Number, Optional, how many pending orders the position has)
}
```
 - *Example:*
```
{
  "action":"account-update",
  "username":"TMBL",
  "BP":1.00,
  "BPU":0.00,
  "CPNL":0.00,
  "OPNL":0.00,
  "investment":0.00,
  "StockPosCnt":0
}
```

### Stock Subscription
 - *Description:* Subscribes to a stock information.
 - *Initiatior:* Mobile Server.
 - *Structure:*
```
{
  "action":"stockL1",
  "username": (String, Username of the user),
  "symbol": (String, Symbol of the stock)
}
```
 - *Example:*
```
{
  "action":"stockL1",
  "username":"TDXX",
  "symbol":"MSFT"
}
```
 - Expected response: see [Stock Snapshot](#stock-snapshot)

### Stock Snapshot
 - *Description:* Responds to a [Stock Subscription](#stock-subscription).
 - *Initiatior:* Middleware.
 - *Structure:*
```
{
  "action":"stockL1-snapshot",
  "symbol": (String, Symbol of the stock),
  "exchange": (String, Symbol of the exchange),
  "username": (String, Username of the user),
  "trd": (String, trading state (one letter)",
  "bidL1": (Float, best price to buy stock),
  "askL1": (Float, best price to sell stock),
  "last": (Float, price of last transaction),
  "lastNbbo": (Float, price of the last transaction that was between (and including) prices bidL1 and askL1),
  "closePrice": (Float, closing price of the stock),
  "openPrice": (Float, opening price of the stock),
  "todayClosePrice": (Float, today's closing price of the stock),
  "volume": (Integer, volume of the stock)
}
```
 - *Example:*
```
{
  “action”:“stockL1-snapshot”,
  “symbol”:“MSFT”,
  “bidL1”:210.83,
  “askL1":210.87,
  “volume”:5640924,
  “last”:210.8649,
  “lastNbbo”:210.8649,
  “closePrice”:210.08,
  “openPrice”:211.76,
  “todayClosePrice”:0.00,
  “exchange”:“NSDQ”,
  “trd”:“T”,
  “username”:“TMBL”
}
```

### Stock Update
 - *Description:* Sends update on the changed stock attributes since last update/snapshot. The stock update is sent while there are users [subscribed](#stock-subscription) to the stock.
 - *Initiatior:* Middleware.
 - *Structure:*
```
{
  "action":"stockL1-update",
  "symbol": (String, Symbol of the stock),
  "trd": (String, Optional, trading state (one letter)",
  "bidL1": (Float, Optional, best price to buy stock),
  "askL1": (Float, Optional, best price to sell stock),
  "last": (Float, Optional, price of last transaction),
  "lastNbbo": (Float, Optional, price of the last transaction that was between (and including) prices bidL1 and askL1),
  "closePrice": (Float, Optional, closing price of the stock),
  "openPrice": (Float, Optional, opening price of the stock),
  "todayClosePrice": (Float, Optional, today's closing price of the stock),
  "volume": (Integer, Optional, volume of the stock)
}
```
 - *Example:*
```
{
  "action":"stockL1-update",
  “symbol”:“MSFT”,
  “bidL1”:210.83,
  “askL1":210.87,
  “volume”:5640924,
  “last”:210.8649,
  “lastNbbo”:210.8649,
  “closePrice”:210.08,
  “openPrice”:211.76,
  “todayClosePrice”:0.00,
  “exchange”:“NSDQ”,
  “trd”:“T”,
  “username”:“TMBL”
}
```

### Position Snapshot
 - *Description:* Message sent right after an [Account Request](#account-request).
 - *Initiatior:* Middleware.
 - *Structure:*
```
{
  "action":"position-snapshot",
  "username":(String, Username of the user),
  "symbol":(String, Symbol of the stock),
  "Size":(Float, TBD),
  "Price":(Float, TBD),
  "BPU":(Float, TBD),
  "CPNL":(Float, TBD),
  "OPNL":(Float, TBD),
  "OrderCnt":(Float, TBD)
}
```
 - *Example:*
```
{
  "action":"position-snapshot",
  "username":"TMBL",
  "symbol":"MSFT",
  "Size":-400,
  "Price":123.45,
  "BPU":49380.00,
  "CPNL":-235.67,
  "OPNL":129.62,
  "OrderCnt":0
}
```

### Position Update
 - *Description:* Responds to a [Stock Subscription](#stock-subscription).
 - *Initiatior:* Middleware.
 - *Structure:*
```
{
  "action":"position-update",
  "username":(String, Username of the user),
  "symbol":(String, Symbol of the stock),
  "Size":(Float, Optional, TBD),
  "Price":(Float, Optional, TBD),
  "BPU":(Float, Optional, TBD),
  "CPNL":(Float, Optional, TBD),
  "OPNL":(Float, Optional, TBD),
  "OrderCnt":(Float, Optional, TBD)
}
```
 - *Example:*
```
{
  "action":"position-update",
  "username":"TMBL",
  "symbol":"MSFT",
  "Size":-400,
  "Price":123.45,
  "BPU":49380.00,
  "CPNL":-235.67,
  "OPNL":129.62,
  "OrderCnt":0
}
```

### Stock Reload
 - *Description:* Like [Stock Snapshot](#stock-snapshot) but to all users (it is used when there has been an error in the connection so that all subscribers must reload their data).
 - *Initiatior:* Middleware.
 - *Structure:*
```
{
  "action":"stockL1-reload",
  "symbol": (String, Symbol of the stock),
  "exchange": (String, Symbol of the exchange),
  "trd": (String, trading state (one letter)",
  "bidL1": (Float, best price to buy stock),
  "askL1": (Float, best price to sell stock),
  "last": (Float, price of last transaction),
  "lastNbbo": (Float, price of the last transaction that was between (and including) prices bidL1 and askL1)
}
```
 - *Example:*
```
{
  "action":"stockL1-reload",
  "symbol":"MSFT",
  "exchange":"NYSE",
  "trd":"A",
  "bidL1":70.21,
  "askL1":70.23,
  "last":70.22,
  "lastNbbo":70.22
}
```

### Unsubscribe to Stock
 - *Description:* Stops subscription to a stock information for a user.
 - *Initiatior:* Mobile Server.
 - *Structure:*
```
{
  "action":"unStockL1",
  "username": (String, Username of the user),
  "symbol": (String, Symbol of the stock)
}
```
 - *Example:*
```
{
  "action":"unStockL1",
  "username":"TDXX",
  "symbol":"MSFT"
}
```
### List of Symbols Request
 - *Description:* Requests a lists of all symbols grouped by provider.
 - *Initiatior:* Mobile Server.
 - *Structure:*
```
{
  "action":"stockList"
}
```
 - *Example:*
```
{
  "action":"stockList"
}
```
 - Expected response: see [List of Symbols](#list-of-symbols)

### List of Symbols
 - *Description:* Lists all symbols grouped by provider, can be either a response to stock list or after a [handshake](#handshake).
 - *Initiatior:* Middleware.
 - *Structure:*
```
{
  "action":"stockList-response",
  PROVIDER_1: (String, list of symbol;name separated by "|"),
  PROVIDER_2: (String, list of symbol;name separated by "|"),
  ...
  PROVIDER_N: (String, list of symbol;name separated by "|"),
}
```
 - *Example:*
```
{
  "action":"stockList-response",
  "ARCA":"TEQI;Tequilom|URE;UrenyPL|XCOM;X Commerce|SPLG;Sole Plea Leverage Goods",
  "TEST_BATS":"ZBZX;Zimb|ZTEST;Ztest",
  "TEST_AMEX":"ATEST.C;Amex Test|ATEST.B; Amex test B|ATEST.A;Amex Test B"
}
```
