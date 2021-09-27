docker kill $(docker ps -q)
docker rm $(docker ps -a -q)
docker image rm takion_trader_receive_market_data
docker image rm takion_trader_receive_order_related_data
docker image rm takion_trader_inference_runner
#docker image rm takion_trader_receive_news
docker image rm redis