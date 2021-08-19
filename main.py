import sys
import asyncio
import traceback
from takion_interaction.tcp_broker import start_interaction

if __name__ == "__main__":
    host = ''
    try:
        host = str(sys.argv[1])
    except Exception as e:
        print(f'Inside main client, '
              f'An exception of type {type(e).__name__}. Arguments: '
              f'{e.args}')
        print(traceback.format_exc())
        host = '127.0.1.1'
    asyncio.run(start_interaction(host))
