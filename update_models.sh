cd ~/takion_trader
conda activate takion_trader
python analytics/modeling/training/update_daily_data.py
python analytics/modeling/training/trainer.py
python analytics/modeling/training/update_minute_data.py
python analytics/modeling/training/create_training_data.py