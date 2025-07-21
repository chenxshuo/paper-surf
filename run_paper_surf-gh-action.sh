source venv/bin/activate
source setup_envs.sh
# Get current day of week (1-7, where 1 is Monday)
DAY_OF_WEEK=$(date +%u)

# Set lookback_days based on day of week
# Monday (1), Tuesday (2): lookback_days=4
# Wednesday (3), Thursday (4), Friday (5): lookback_days=2
# Weekend (6,7): use default from config
if [ "$DAY_OF_WEEK" -eq 1 ] || [ "$DAY_OF_WEEK" -eq 2 ]; then
    # Monday or Tuesday
    LOOKBACK_DAYS=4
    echo "Running on Monday/Tuesday with lookback_days=$LOOKBACK_DAYS"
    ./venv/bin/python run_daily.py  --lookback-days $LOOKBACK_DAYS 
elif [ "$DAY_OF_WEEK" -ge 3 ] && [ "$DAY_OF_WEEK" -le 5 ]; then
    # Wednesday, Thursday, or Friday
    LOOKBACK_DAYS=2
    echo "Running on Wed/Thu/Fri with lookback_days=$LOOKBACK_DAYS"
    ./venv/bin/python run_daily.py --lookback-days $LOOKBACK_DAYS
else
    # Weekend - use default from config
    echo "Happy Weekend!"
fi