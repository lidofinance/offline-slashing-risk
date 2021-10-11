import numpy as np
import requests
import logging
import math
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format



SLOTS_PER_EPOCH = 32
BASE_REWARDS_PER_EPOCH = 4
BASE_REWARD_FACTOR = 2**6
MAX_EFFECTIVE_BALANCE = 2**5 * 10**9
EFFECTIVE_BALANCE_INCREMENT = 2**0 * 10**9
EPOCHS_PER_SLASHING = 2**13 # ~36 days or 8192 epochs
MIN_SLASHING_PENALTY_QUOTIENT = 2**7
MIN_SLASHING_PENALTY_QUOTIENT_ALTAIR = 2**6 # (= 64)
PROPORTIONAL_SLASHING_MULTIPLIER = 1
PROPORTIONAL_SLASHING_MULTIPLIER_ALTAIR	= 2
TIMELY_SOURCE_WEIGHT=14
TIMELY_TARGET_WEIGHT=26
TIMELY_HEAD_WEIGHT=14
SYNC_REWARD_WEIGHT=2
PROPOSER_WEIGHT=8
WEIGHT_DENOMINATOR=64
SYNC_COMMITTEE_SIZE=2**9 # (= 512)	Validators	
EPOCHS_PER_SYNC_COMMITTEE_PERIOD=2**8 # (= 256)	epochs	~27 hours
HYSTERESIS_QUOTIENT = 4
HYSTERESIS_DOWNWARD_MULTIPLIER = 1
HYSTERESIS_UPWARD_MULTIPLIER = 5
HYSTERESIS_INCREMENT = EFFECTIVE_BALANCE_INCREMENT // HYSTERESIS_QUOTIENT
DOWNWARD_THRESHOLD = HYSTERESIS_INCREMENT * HYSTERESIS_DOWNWARD_MULTIPLIER
UPWARD_THRESHOLD = HYSTERESIS_INCREMENT * HYSTERESIS_UPWARD_MULTIPLIER
MAX_VALIDATOR_COUNT = 2**19 # (= 524,288)



## Get current beacon chain data

def get_epoch_data(epoch="latest"):
    try:
        req = requests.get(f"https://beaconcha.in/api/v1/epoch/{epoch}", headers={"accept":"application/json"})
        req.raise_for_status()
        return req.json()["data"]
    except requests.exceptions.HTTPError as err:
        logging.error(err)
        return {}



## Get aggregated results

def get_results_offline(exams, inputdata, epochs_offline):
    states = ['current', 'future']
    specs = ['Phase 0', 'Altair']
    results = [get_result_offline(epochs_offline, exams, inputdata, state, spec) for spec in specs for state in states]
    titles = [(state, spec) for spec in specs for state in states]
    for result in range(len(results)):
        pd.options.display.float_format = '{:,.2f}'.format
        display(titles[result], str(epochs_offline)+' epochs_offline')
        display(results[result])
        print()
    #return results

def get_results_slashing(exams, inputdata):
    states = ['current', 'future']
    specs = ['Phase 0', 'Altair']
    results = [get_result_slashing(exams, inputdata, state, spec) for spec in specs for state in states]
    titles = [(state, spec) for spec in specs for state in states]
    for result in range(len(results)):
        pd.options.display.float_format = '{:,.2f}'.format
        display(titles[result])
        display(results[result])
        print()
    #return results



## Get result for a given state (current, future) and spec (Phase 0, Altair)

def get_result_offline(epochs_offline, exams, inputdata, state, spec):

    if state == 'future': x = 1
    else: x = 0

    lidoavgbalance = inputdata.get("avarage balance of Lido's validators")[x]*EFFECTIVE_BALANCE_INCREMENT
    lidoavgeffbalance = inputdata.get("avarage effective balance of Lido's validators")[x]*EFFECTIVE_BALANCE_INCREMENT
    validatorscount = inputdata.get('total active validators')[x]
    avarage_effective_balance = inputdata.get('total eligible ETH')[x]*EFFECTIVE_BALANCE_INCREMENT/validatorscount
    lidostakeddeposits = inputdata.get("Lido's deposits")[x]*EFFECTIVE_BALANCE_INCREMENT
    lidotreasury = inputdata.get("5y_earnings")[x]*EFFECTIVE_BALANCE_INCREMENT

    result_list = [get_exam_offline(
        epochs_offline, exams[y][x], lidoavgbalance, lidoavgeffbalance, validatorscount, avarage_effective_balance, spec
        ) for y in range(len(exams))]

    df_result = pd.DataFrame(
        result_list, 
        index = [
            'single big operator, 30% validators offline',
            'single big operator, 100% validators offline',
            'two big operators, 30% validators offline',
            'two big operators, 100% validators offline'])
    df_result['%_of_lido_deposits'] = df_result.total_loss/gwei_to_ether(lidostakeddeposits)
    df_result['%_of_5y_earnings'] = df_result.total_loss/gwei_to_ether(lidotreasury)
    return df_result[['total_loss','%_of_lido_deposits','%_of_5y_earnings']]

def get_result_slashing(exams, inputdata, state, spec):

    if state == 'future': x = 1
    else: x = 0

   
    lidoavgbalance = inputdata.get("avarage balance of Lido's validators")[x]*EFFECTIVE_BALANCE_INCREMENT
    lidoavgeffbalance = inputdata.get("avarage effective balance of Lido's validators")[x]*EFFECTIVE_BALANCE_INCREMENT
    validatorscount = inputdata.get('total active validators')[x]
    avarage_effective_balance = inputdata.get('total eligible ETH')[x]*EFFECTIVE_BALANCE_INCREMENT/validatorscount
    lidostakeddeposits = inputdata.get("Lido's deposits")[x]*EFFECTIVE_BALANCE_INCREMENT
    lidotreasury = inputdata.get("5y_earnings")[x]*EFFECTIVE_BALANCE_INCREMENT

    result_list = [get_exam_slashing(
        exams[y][x], lidoavgbalance, lidoavgeffbalance, validatorscount, avarage_effective_balance, spec
        ) for y in range(len(exams))]

    df_result = pd.DataFrame(
        result_list, 
        index = [
            'single big operator, 30% validators slashed',
            'single big operator, 100% validators slashed',
            'two big operators, 30% validators slashed',
            'two big operators, 100% validators slashed'])
    df_result['%_of_lido_deposits'] = df_result.total_loss/gwei_to_ether(lidostakeddeposits)
    df_result['%_of_5y_earnings'] = df_result.total_loss/gwei_to_ether(lidotreasury)
    return df_result[['total_loss','%_of_lido_deposits','%_of_5y_earnings']]



## Get result for a given exam

def get_exam_offline(epochs_offline, exam, lidoavgbalance, lidoavgeffbalance, validatorscount, avarage_effective_balance, spec):
    dic = {}
    if spec == 'Altair': 
        result = process_offline_validator_altair(epochs_offline, lidoavgbalance, lidoavgeffbalance, validatorscount, avarage_effective_balance) 
    else:
        result = process_offline_validator(epochs_offline, lidoavgbalance, lidoavgeffbalance, validatorscount, avarage_effective_balance)

    prob_number_validators_assigned = get_probability_outcomes(exam, validatorscount)*result[4]
    dic.update({'offline_count': exam})
    dic.update({'total_loss_offline_penalty': gwei_to_ether((result[0]-lidoavgbalance)*exam)})
    dic.update({'average_loss_offline_penalty': gwei_to_ether(result[0]-lidoavgbalance)})
    dic.update({'total_loss': gwei_to_ether((result[0]-lidoavgbalance)*(exam-prob_number_validators_assigned)+(result[2]-lidoavgbalance)*prob_number_validators_assigned)})
    dic.update({'average_loss': gwei_to_ether(result[2]-lidoavgbalance)})
    return dic

def get_exam_slashing(exam, lidoavgbalance, lidoavgeffbalance, validatorscount, avarage_effective_balance, spec):
    dic = {}
    if spec == 'Altair': 
        result = process_slashings_altair(exam, lidoavgbalance, lidoavgeffbalance, validatorscount, avarage_effective_balance)   
    else:
        result = process_slashings(exam, lidoavgbalance, lidoavgeffbalance, validatorscount, avarage_effective_balance)
    dic.update({'slashings_count': exam})
    dic.update({'total_loss': gwei_to_ether((result[0]-lidoavgbalance)*exam)})
    dic.update({'average_loss': gwei_to_ether(result[0]-lidoavgbalance)})
    return dic



## Offline penalties calculation Phase 0

def process_offline_validator(epochs_offline, lidoavgbalance, lidoavgeffbalance, validatorscount, avarage_effective_balance):
    balance = lidoavgbalance
    effective_balance = lidoavgeffbalance
    total_active_balances = (validatorscount)* avarage_effective_balance
    epoch = 0
    while epoch<= epochs_offline:
        balance, effective_balance = process_offline_penalty(balance, effective_balance, total_active_balances)
        epoch += 1
    return balance, effective_balance, balance, effective_balance, 0


## Offline penalties calculation Altair

def process_offline_validator_altair(epochs_offline, lidoavgbalance, lidoavgeffbalance, validatorscount, avarage_effective_balance):
    sync_comittees = math.ceil(epochs_offline/EPOCHS_PER_SYNC_COMMITTEE_PERIOD)
    sync_penalty_period = sync_comittees*EPOCHS_PER_SYNC_COMMITTEE_PERIOD
    balance = lidoavgbalance
    balance_sync = lidoavgbalance
    effective_balance = lidoavgeffbalance
    effective_balance_sync = lidoavgeffbalance
    total_active_balances = (validatorscount)* avarage_effective_balance
    epoch = 0
    while epoch<= epochs_offline:
        balance, effective_balance = process_offline_penalty_altair(balance, effective_balance, total_active_balances)
        balance_sync, effective_balance_sync = process_offline_penalty_altair(balance_sync, effective_balance_sync, total_active_balances)
        balance_sync, effective_balance_sync = process_sync_penalty_altair(balance_sync, effective_balance_sync, total_active_balances)
        epoch += 1
    if epochs_offline < sync_penalty_period:
        while epoch<=sync_penalty_period:
            balance_sync, effective_balance_sync = process_sync_penalty_altair(balance_sync, effective_balance_sync, total_active_balances)
            epoch += 1
    
    return balance, effective_balance, balance_sync, effective_balance_sync, sync_comittees



## Slashing penalties calculation Phase 0

def process_slashings(slashed_validator_cnt, lidoavgbalance, lidoavgeffbalance, validatorscount, avarage_effective_balance):
    balance = lidoavgbalance
    effective_balance = lidoavgeffbalance
    total_active_balances = (validatorscount - slashed_validator_cnt)* avarage_effective_balance
    slashed_validator_balance = effective_balance * slashed_validator_cnt

    #initial_penalty
    balance, effective_balance = process_initial_penalty(balance, effective_balance)
    
    #offline_penalty
    exiting_epoch = 0
    while exiting_epoch <= EPOCHS_PER_SLASHING//2:
        balance, effective_balance = process_offline_penalty(balance, effective_balance, total_active_balances)
        exiting_epoch += 1
    
    # special_penalty
    balance, effective_balance = process_special_penalty(balance, effective_balance, total_active_balances, slashed_validator_balance)

    #offline_penalty
    while exiting_epoch <= EPOCHS_PER_SLASHING:
        balance, effective_balance = process_offline_penalty(balance, effective_balance, total_active_balances)
        exiting_epoch += 1

    return balance, effective_balance



## Slashing penalties calculation Altair

def process_slashings_altair(slashed_validator_cnt, lidoavgbalance, lidoavgeffbalance, validatorscount, avarage_effective_balance):
    balance = lidoavgbalance
    effective_balance = lidoavgeffbalance
    total_active_balances = (validatorscount - slashed_validator_cnt)* avarage_effective_balance
    slashed_validator_balance = effective_balance * slashed_validator_cnt

    #initial_penalty
    balance, effective_balance = process_initial_penalty_altair(balance, effective_balance)
    
    #exiting_penalty
    exiting_epoch = 0
    while exiting_epoch <= EPOCHS_PER_SLASHING//2:
        balance, effective_balance = process_offline_penalty_altair(balance, effective_balance, total_active_balances)
        exiting_epoch += 1
    
    # special_penalty
    balance, effective_balance = process_special_penalty_altair(balance, effective_balance, total_active_balances, slashed_validator_balance)

    #exiting_penalty
    while exiting_epoch <= EPOCHS_PER_SLASHING:
        balance, effective_balance = process_offline_penalty_altair(balance, effective_balance, total_active_balances)
        exiting_epoch += 1

    return balance, effective_balance



## Penalties Phase 0

def process_initial_penalty(balance, effective_balance):
    initial_penalty = effective_balance // MIN_SLASHING_PENALTY_QUOTIENT
    balance -= initial_penalty
    effective_balance = process_final_updates(balance, effective_balance)
    return balance, effective_balance

def process_offline_penalty(balance, effective_balance, total_active_balances):
    base_reward = (effective_balance * BASE_REWARD_FACTOR // integer_squareroot(total_active_balances)) //BASE_REWARDS_PER_EPOCH
    offline_penalty = 3 * base_reward
    balance -= offline_penalty
    effective_balance = process_final_updates(balance, effective_balance)
    return balance, effective_balance

def process_special_penalty(balance, effective_balance, total_active_balances, slashed_validator_balance):
    special_penalty = effective_balance * min(
                    slashed_validator_balance*PROPORTIONAL_SLASHING_MULTIPLIER, total_active_balances) // total_active_balances 
    balance -= special_penalty
    effective_balance = process_final_updates(balance, effective_balance)
    return balance, effective_balance



## Penalties Altair

def process_initial_penalty_altair(balance, effective_balance):
    initial_penalty = effective_balance // MIN_SLASHING_PENALTY_QUOTIENT_ALTAIR
    balance -= initial_penalty
    effective_balance = process_final_updates(balance, effective_balance)
    return balance, effective_balance

def process_offline_penalty_altair(balance, effective_balance, total_active_balances):
    base_reward = effective_balance // EFFECTIVE_BALANCE_INCREMENT * (EFFECTIVE_BALANCE_INCREMENT*BASE_REWARD_FACTOR // integer_squareroot(total_active_balances)) 
    offline_penalty = (TIMELY_SOURCE_WEIGHT+TIMELY_TARGET_WEIGHT+TIMELY_HEAD_WEIGHT)/WEIGHT_DENOMINATOR * base_reward
    balance -= offline_penalty
    effective_balance = process_final_updates(balance, effective_balance)
    return balance, effective_balance

def process_special_penalty_altair(balance, effective_balance, total_active_balances, slashed_validator_balance):
    special_penalty = effective_balance * min(
                    slashed_validator_balance*PROPORTIONAL_SLASHING_MULTIPLIER_ALTAIR, total_active_balances) // total_active_balances 
    balance -= special_penalty
    effective_balance = process_final_updates(balance, effective_balance)
    return balance, effective_balance

def process_sync_penalty_altair(balance, effective_balance, total_active_balances):
    total_active_increments = total_active_balances // EFFECTIVE_BALANCE_INCREMENT
    total_base_rewards = EFFECTIVE_BALANCE_INCREMENT*BASE_REWARD_FACTOR // integer_squareroot(total_active_balances) * total_active_increments
    max_participant_rewards = total_base_rewards * SYNC_REWARD_WEIGHT // WEIGHT_DENOMINATOR // SLOTS_PER_EPOCH
    participant_reward = max_participant_rewards // SYNC_COMMITTEE_SIZE
    balance -= participant_reward
    effective_balance = process_final_updates(balance, effective_balance)
    return balance, effective_balance



## Final updates

def process_final_updates(balance, effective_balance):
        # Update effective balances with hysteresis
        if (
            balance + DOWNWARD_THRESHOLD < effective_balance
            or effective_balance + UPWARD_THRESHOLD < balance
        ):
            effective_balance = min(balance - balance % EFFECTIVE_BALANCE_INCREMENT, MAX_EFFECTIVE_BALANCE)
        return effective_balance



## Helpers

def gwei_to_ether(amount):
    return amount/10**9

def integer_squareroot(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def c(n, k):   # helper for large numbers binomial coefficient calculation
    if 0 <= k <= n:
        nn = 1
        kk = 1
        for t in range(1, min(k, n - k) + 1):
            nn *= n
            kk *= t
            n -= 1
        return nn // kk
    else:
        return 0

def get_probability_outcomes(exam, validatorscount):
    outcome = []
    for offline_validator_sync_cnt in range(0, SYNC_COMMITTEE_SIZE+1):
        outcome.append(c(int(exam), offline_validator_sync_cnt)*c(int(validatorscount-exam),SYNC_COMMITTEE_SIZE-offline_validator_sync_cnt)/c(int(validatorscount),SYNC_COMMITTEE_SIZE))
    df_outcome = pd.DataFrame(pd.Series(outcome), columns=['outcome'])
    df_outcome['cumul'] = df_outcome.outcome.cumsum()
    return df_outcome[df_outcome.cumul <= 0.99].tail(1).index[0]



## MAIN

# current state of beacon chain
current_epoch_data = get_epoch_data()
validatorscount_current = int(current_epoch_data['validatorscount'])
current_epoch = current_epoch_data['epoch']
totalvalidatorbalance_current = current_epoch_data['totalvalidatorbalance']
eligibleether_current=current_epoch_data['eligibleether']
avarage_effective_balance = eligibleether_current/validatorscount_current
validatorscount_current_param = validatorscount_current*1.05-validatorscount_current*1.05%10000
# params used in the calculation for the publication
#validatorscount_current = 241558
#current_epoch = 66444
#totalvalidatorbalance_current = 7995859764391736
#eligibleether_current = 7729659000000000

# Lido param
current_lido_deposits = 1350000
current_lido_treasury = 10.154232576699999999*365*5 # 5 year Lido earnings at current rate
future_lido_treasury = 10.154232576699999999*365*5 # 5 year Lido earnings at current rate
future_lido_share = 0.25
current_lido_validator_average_eff_balance = MAX_EFFECTIVE_BALANCE
current_lido_validator_average_balance = 33000000000
future_lido_validator_average_eff_balance = MAX_EFFECTIVE_BALANCE
future_lido_validator_average_balance = 33000000000
current_biggest_node_operator = 5500
future_biggest_node_operator = 10000
period_offline = 256 #epochs




# params for calculation
validatorscount = np.array([validatorscount_current_param, MAX_VALIDATOR_COUNT])#validatorscount_current_param*10])#MAX_VALIDATOR_COUNT])
eligibleether = validatorscount*(gwei_to_ether(avarage_effective_balance))
lidoshare = [current_lido_deposits/gwei_to_ether(current_epoch_data['eligibleether']),future_lido_share]
lidotreasury = [current_lido_treasury, future_lido_treasury]
lidostakeddeposits = [eligibleether[x]*lidoshare[x] for x in range(len(lidoshare))]
lidoavgeffbalance = gwei_to_ether(np.array([current_lido_validator_average_eff_balance,future_lido_validator_average_eff_balance]))
lidoavgbalance = gwei_to_ether(np.array([current_lido_validator_average_balance,future_lido_validator_average_balance]))
lidobigoperator = [[current_biggest_node_operator, current_biggest_node_operator], [future_biggest_node_operator,future_biggest_node_operator]]
exams = [
    [0.3*lidobigoperator[0][0], 0.3*lidobigoperator[1][0]],
    [lidobigoperator[0][0], lidobigoperator[1][0]],
    [0.3*sum(lidobigoperator[0]),0.3*sum(lidobigoperator[1])],
    [sum(lidobigoperator[0]),sum(lidobigoperator[1])]
]
inputdata = {
        'total active validators':validatorscount,
        'total eligible ETH':eligibleether,
        "Lido's share":lidoshare,
        "Lido's deposits":lidostakeddeposits,
        "5y_earnings":lidotreasury,
        "avarage effective balance of Lido's validators":lidoavgeffbalance,
        "avarage balance of Lido's validators":lidoavgbalance,
        'single big operator, 30% validators offline/slashed':exams[0],
        'single big operator, 100% validators offline/slashed':exams[1],
        'two big operators, 30% validators offline/slashed':exams[2],
        'two big operators, 100% validators offline/slashed':exams[3]
}
df_inputdata = pd.DataFrame(
    inputdata
    ).T
df_inputdata.columns=['current_state', 'future_state']

print(df_inputdata)
print()

# offline penalties modeling
print("offline penalties modeling")
get_results_offline(exams, inputdata, period_offline)

# slashing penalties modeling
print("slashing penalties modeling")
get_results_slashing(exams, inputdata)