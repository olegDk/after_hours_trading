default="no"
restart="yes"
read -e -p "Do you want to restart docker services? ([no]/yes): " -i $default VAR
if [[ $VAR = $restart ]]; then
  source prod_run/stop_rmi.sh
  docker-compose up -d -V
fi
source prod_run/start_client.sh