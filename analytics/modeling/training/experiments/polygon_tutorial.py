from datetime import datetime, timedelta
from pytz import timezone
from polygon import RESTClient

EST = timezone('US/Eastern')
key = 'EuMC51EF6YZypR2x5kS__K72C6FKdwhF'


def ts_to_datetime(ts: int) -> str:
    return (datetime.fromtimestamp(ts / 1000.0, tz=EST)).strftime('%Y-%m-%d %H:%M')


key = 'EuMC51EF6YZypR2x5kS__K72C6FKdwhF'

# RESTClient can be used as a context manager to facilitate closing the underlying http session
# https://requests.readthedocs.io/en/master/user/advanced/#session-objects
with RESTClient(key) as client:
    from_ = '2021-12-10'
    to = '2021-12-10'
    resp = client.stocks_equities_aggregates('C', 1, 'minute', from_, to, unadjusted=False)

    print(f'5 minute aggregates for {resp.ticker} between {from_} and {to}.')

    for result in resp.results:
        dt = ts_to_datetime(result['t'])
        print(f"{dt}\n\tO: {result['o']}\n\tH: {result['h']}\n\tL: {result['l']}\n\tC: {result['c']} ")


# with RESTClient(key) as client:
#     resp = client.stocks_equities_daily_open_close(symbol='TSLA',
#                                                    date='2021-11-10')
# resp.__getattribute__('close')

# with RESTClient(key) as client:
#     resp = client.reference_tickers_v3(active=True,
#                                        market='stocks',
#                                        type='CS',
#                                        limit=1000)

resp.results[-1]

ts_to_datetime(ts=1639184340000)
