Assume the user has at least one week's historical data:

Runs when the user syncs data:
1. prepare_fitbit_data.py
2. generate_pa_display.py
3. generate_pa_plan.py
4. generate_sedentary_break_plan.py

Runs everyday at 5am:
1. generate_pa_plan.py
2. generate_sedentary_break_plan.py

When user do not have at least one week's historical data:
1. prepare_fitbit_data.py
2. generate_pa_display.py

Python Dependencies:
1. pandas
2. numpy
3. json
4. sklearn
5. scipy