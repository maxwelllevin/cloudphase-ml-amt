# Utility script to place an order from arm.gov for nsathermocldphaseC1.c1 data from
# 2018 through 2021 for use in this paper.
#
# Output data will be placed in a folder named 'nsathermocldphaseC1.c1'.
#
# Prerequisites:
# 1. Create an ARM.gov account (free): https://adc.arm.gov/armuserreg/#/new
# 2. Get your API token by logging in to https://adc.arm.gov/armlive/
# 3. Install act-atmos (https://arm-doe.github.io/ACT/)
#

import act  # "conda install act-atmos" OR "pip install act-atmos"

# Set your username and token here!
username = "YourUserName"
token = "YourToken"


files = act.discovery.download_arm_data(
    username=username,
    token=token,
    datastream="nsathermocldphaseC1.c1",
    startdate="2018-01-01",
    enddate="2022-01-01",
)
