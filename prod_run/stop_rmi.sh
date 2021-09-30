docker kill $(docker ps -q)
docker rm $(docker ps -a -q)
docker image rm algotrading_receive_market_data
docker image rm algotrading_receive_order_related_data
docker image rm algotrading_inference_runner
#docker image rm algotrading_receive_news
docker image rm redis
docker image rm rabbitmq:3.8-management
docker volume rm $(docker volume ls -f dangling=true -q)