import asyncio
from takion_interaction.tcp_broker import start_interaction

if __name__ == "__main__":
    # execute only if run as a script
   asyncio.run(start_interaction())
