default="no"
start="yes"
read -e -p "Do you want to start local server? ([no]/yes): " -i $default VAR
if [[ $VAR = $start ]]; then
  cd /home/takion_trader/algotrading
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate takion_trader
  python emulator/server.py
  conda deactivate
  cd ~
fi
