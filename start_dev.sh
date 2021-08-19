default="no"
restart="yes"
read -e -p "Do you want to restart docker services? ([no]/yes): " -i $default VAR
if [[ $VAR = $restart ]]; then
  source dev_run/stop_rmi_dev.sh
  gnome-terminal --working-directory='~/takion_trader' -- bash -c "cd ~/takion_trader; docker-compose up; exec bash"
fi
gnome-terminal --working-directory='~/takion_trader' -- bash -c "cd ~/takion_trader; source dev_run/start_server_dev.sh; exec bash"
gnome-terminal --working-directory='~/takion_trader' -- bash -c "cd ~/takion_trader; source dev_run/start_client_dev.sh; exec bash"
