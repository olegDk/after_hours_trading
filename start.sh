default="no"
restart="yes"
read -e -p "Do you want to restart docker services? ([no]/yes): " -i $default VAR
if [[ $VAR = $restart ]]; then
  source prod_run/stop_rmi.sh
  gnome-terminal --working-directory='~/takion_trader' -- bash -c "cd ~/takion_trader; docker-compose up; exec bash"
fi
gnome-terminal --working-directory='~/takion_trader' -- bash -c "cd ~/takion_trader; source prod_run/start_client.sh; exec bash"