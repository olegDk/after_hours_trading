default="no"
restart="yes"
read -e -p "Do you want to restart docker services? ([no]/yes): " -i $default VAR
if [[ $VAR = $restart ]]; then
  source remote_dev_run/stop_rmi_remote_dev.sh
  gnome-terminal --working-directory='/home/takion_trader/algotrading' -- bash -c "cd /home/takion_trader/algotrading; docker-compose up; exec bash"
fi
gnome-terminal --working-directory='/home/takion_trader/algotrading' -- bash -c "cd /home/takion_trader/algotrading; source remote_dev_run/start_server_remote_dev.sh; exec bash"
gnome-terminal --working-directory='/home/takion_trader/algotrading' -- bash -c "cd /home/takion_trader/algotrading; source remote_dev_run/start_client_remote_dev.sh; exec bash"