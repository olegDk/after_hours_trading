cd ~/takion_trader
source ~/miniconda3/etc/profile.d/conda.sh
conda activate takion_trader
dev="no"
prod="yes"
read -e -p "Starting client in dev mode? ([no]/yes): " -i $dev VAR
if [[ $VAR = $prod ]]; then
  python main.py '127.0.1.1'
  conda deactivate
  cd ~
else
  python main.py '192.168.137.11'
  conda deactivate
  cd ~
fi
