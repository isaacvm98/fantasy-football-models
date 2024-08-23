# get replacement level player by position
import pandas as pd
import numpy as np
from scipy.stats import norm
import requests

# league rules!
numberOfTeams = 16
isFlex = True
ppr = 1
rushYards = .1
receivingYards = .1
rushTD = 6
receivingTD = 6
passYards = 1/25
passTD = 4
fmb = -2
int = -2

# get data
individuals = pd.read_csv('data/individuals.csv')
preds_copy = pd.read_csv('data/preds_copy.csv')

ntrees = 500


# your team
draftedOverall = []

yourTeam = []

fuzzy_replacements = {'7547': 'AmonRaSt',
                    '7670': 'JoshPalmer',
                    '166': 'JoshuaCribbs',
                    '6290': 'ScottMiller',
                    '226': 'AntwaanRandle',
                    '308': 'JohnnieLee',
                    '5773': 'KhadarelHodge',
                    '2247': 'WaltPowell',
                    '3384': 'DeMarcusAyers',
                    '5539': 'DerrickWilliams',
                    '8917': 'KavontaeTurpin',
                    '169': 'SteveSmith2',
                    '108': 'StevieJohnson',
                    '9501': 'DemarioDouglas',
                    '6996': 'JamycalHasty',
                    '7863': 'JaquanHardy',
                    '5052': 'RonaldJonesII',
                    '3668': 'JoshPerkins',
                    '12357': 'DavidMartin',
                    '761': 'BrianSt'}

def replace_names(drafted_df, fuzzy_replacements):
    # Create a function to replace names
    def replace_name(row):
        sleeper_id = str(row['sleeper_id'])  # Convert to string to match dictionary keys
        if sleeper_id in fuzzy_replacements:
            return fuzzy_replacements[sleeper_id]
        else:
            return row['Name']

    # Apply the function to create a new 'Name' column
    drafted_df['Name'] = drafted_df.apply(replace_name, axis=1)

    return drafted_df
# roundup function
def roundUp(x, to=numberOfTeams):
    return to * (x // to + (x % to > 0))

def get_sleeper_draft_picks(draft_id):
    url = f"https://api.sleeper.app/v1/draft/{draft_id}/picks"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching draft picks: {response.status_code}")
        return None

def update_drafted_players(draft_picks,user_id):
    draft_info = []
    for pick in draft_picks:
        picked_by = pick.get('picked_by')
        metadata = pick.get('metadata', {})
        sleeper_id = pick.get('player_id',{})
        draft_info.append({
            'first_name': f"{metadata.get('first_name', '')}",
            'last_name':f"{metadata.get('last_name', '')}",
            'picked_by':f"{picked_by}",
            'sleeper_id':f"{sleeper_id}"
        })
    drafted_df = pd.DataFrame(draft_info)
    drafted_df['first_name'] = drafted_df['first_name'].str.replace(r'[^a-zA-Z]', '', regex=True)
    drafted_df['last_name'] = drafted_df['last_name'].str.replace(r'[^a-zA-Z]', '', regex=True)
    drafted_df['Name'] = drafted_df['first_name'] + drafted_df['last_name']
    drafted_df = replace_names(drafted_df, fuzzy_replacements)
    draftedOverall = drafted_df['Name'].to_list()
    yourTeam = drafted_df[drafted_df['picked_by']==user_id]['Name'].to_list()
    return draftedOverall,yourTeam

def get_draft_info(draft_picks):
    draft_info = []
    for pick in draft_picks:
        metadata = pick.get('metadata', {})
        draft_info.append({
            'first_name': f"{metadata.get('first_name', '')}",
            'last_name':f"{metadata.get('last_name', '')}",
            'position': metadata.get('position', ''),
            'team': metadata.get('team', ''),
            'pick_no': pick.get('pick_no', ''),
            'round': pick.get('round', ''),
            'draft_slot': pick.get('draft_slot', ''),
            'roster_id': pick.get('roster_id', ''),
            'is_keeper': pick.get('is_keeper', False)
        })
    draft_info = pd.DataFrame(draft_info)
    return draft_info

def draft_optimize(ppr, num_teams, draft_id, user_id):
    # Fetch the latest draft picks from Sleeper API
    latest_picks = get_sleeper_draft_picks(draft_id)
    if latest_picks:
        draftedOverall,yourTeam = update_drafted_players(latest_picks, user_id)
        draft_info_df = get_draft_info(latest_picks)
        print("Current Draft Status:")
        print(draft_info_df)

        print('\n')
        print('Current team:')
        print(yourTeam)
    else:
        print("Failed to fetch latest draft picks. Using existing draftedOverall list.")

    # Load data based on PPR setting
    if ppr == '0':
        individuals_path = "data/individuals_0_PPR.csv"
        preds_copy_path = "data/preds_copy_0_PPR.csv"
    elif ppr == '0.5':
        individuals_path = "data/individuals_.5_PPR.csv"
        preds_copy_path = "data/preds_copy_.5_PPR.csv"
    else:
        individuals_path = "data/individuals_1_PPR.csv"
        preds_copy_path = "data/preds_copy_1_PPR.csv"

    individuals = pd.read_csv(individuals_path)
    preds_copy = pd.read_csv(preds_copy_path)

    # Filtering data based on conditions
    replacement_rb = individuals[(individuals['pos'] == 'RB') & (
        individuals['posrank'] == (num_teams*2.5 + 1))].iloc[:, :ntrees]
    replacement_wr = individuals[(individuals['pos'] == 'WR') & (
        individuals['posrank'] == (num_teams*2.5 + 1))].iloc[:, :ntrees]
    replacement_te = individuals[(individuals['pos'] == 'TE') & (
        individuals['posrank'] == (num_teams + 1))].iloc[:, :ntrees]
    replacement_qb = individuals[(individuals['pos'] == 'QB') & (
        individuals['posrank'] == (num_teams + 1))].iloc[:, :ntrees]

    # Initialize DataFrames and lists
    createdDataframe = pd.DataFrame()
    secondDataframe = pd.DataFrame()
    created_rows = []
    yourDraft = individuals[individuals['name'].isin(yourTeam)]
    yourDraft['teamrank'] = np.int64(yourDraft.groupby(
        'pos')['preds'].rank(ascending=False, method='min'))

    first_rb = yourDraft[(yourDraft['teamrank'] == 1)
                         & (yourDraft['pos'] == 'RB')]
    second_rb = yourDraft[(yourDraft['teamrank'] == 2)
                          & (yourDraft['pos'] == 'RB')]
    third_rb = yourDraft[(yourDraft['teamrank'] == 3)
                         & (yourDraft['pos'] == 'RB')]
    first_wr = yourDraft[(yourDraft['teamrank'] == 1)
                         & (yourDraft['pos'] == 'WR')]
    second_wr = yourDraft[(yourDraft['teamrank'] == 2)
                          & (yourDraft['pos'] == 'WR')]
    third_wr = yourDraft[(yourDraft['teamrank'] == 3)
                         & (yourDraft['pos'] == 'WR')]
    first_te = yourDraft[(yourDraft['teamrank'] == 1)
                         & (yourDraft['pos'] == 'TE')]
    first_qb = yourDraft[(yourDraft['teamrank'] == 1)
                         & (yourDraft['pos'] == 'QB')]

    first_flex = third_rb if third_rb.preds.max() > third_wr.preds.max() else third_wr

    # For RBs
    t_rb = replacement_rb.transpose()
    rbs = pd.concat([t_rb.sample(frac=1).transpose()
                    for _ in range(3)], axis=0)
    if not first_rb.empty:
        rbs.iloc[0, :] = first_rb.iloc[0, :ntrees]
    if not second_rb.empty:
        rbs.iloc[1, :] = second_rb.iloc[0, :ntrees]
    if not first_flex.empty:
        rbs.iloc[2, :] = first_flex.iloc[0, :ntrees]

    # For WRs
    t_wr = replacement_wr.transpose()
    wrs = pd.concat([t_wr.sample(frac=1).transpose()
                    for _ in range(3)], axis=0)
    if not first_wr.empty:
        wrs.iloc[0, :] = first_wr.iloc[0, :ntrees]
    if not second_wr.empty:
        wrs.iloc[1, :] = second_wr.iloc[0, :ntrees]
    if not first_flex.empty:
        wrs.iloc[2, :] = first_flex.iloc[0, :ntrees]

    # For TEs
    # Create a copy to prevent modification of original dataframe
    tes = replacement_te.copy()
    if not first_te.empty:
        tes.iloc[0, :] = first_te.iloc[0, :ntrees]

    # For QBs
    # Create a copy to prevent modification of original dataframe
    qbs = replacement_qb.copy()
    if not first_qb.empty:
        qbs.iloc[0, :] = first_qb.iloc[0, :ntrees]

    # Dictionary for positions
    pos_dict = {
        'RB': rbs,
        'WR': wrs,
        'TE': tes,
        'QB': qbs
    }

    # Filter out players in draftedOverall from copy
    available = preds_copy[~preds_copy['name'].isin(draftedOverall)]
    available = available.sort_values(by=['pos', 'preds'], ascending=False).groupby(
        'pos').head(25).reset_index(drop=True)

    for j in range(len(available)):
        player = available.iloc[j]
        name = player['name']
        position = player['pos']

        # Use dictionary to get positional_df
        positional_df = pos_dict[position]

        indi_preds = individuals[(individuals['name'] == name) & (
            individuals['pos'] == position)].iloc[:, :ntrees].iloc[0]
        total_pt_gains = 0
        pct_better = 0

        for i in range(len(positional_df)):
            row = positional_df.iloc[i]
            better = indi_preds.values > row[::-1].values  # Using values to get numpy arrays
            elementwise_gains = np.sum(indi_preds[better].values - np.array(row[::-1])[better]) / ntrees
            total_pt_gains = max(total_pt_gains, elementwise_gains)

        # Finding pick number, your next pick, and the likelihood of a player staying on board
        pickNumber = len(draftedOverall) + 1
        ceiling = roundUp(pickNumber, num_teams)
        leftTillEndOfRound = ceiling - pickNumber
        nextPick = ceiling + leftTillEndOfRound + 1

        # Calculating the chance of staying on board
        chanceOfStayingOnBoard = round(
            1 - norm.cdf(nextPick, player['adp'], player['adp_sd']), 2)

        # For the round after the likelihood
        ceiling = roundUp(nextPick, num_teams)
        leftTillEndOfRound = ceiling - nextPick
        pickAfter = ceiling + leftTillEndOfRound + 1

        chanceOfStayingOnBoardTwoRounds = 1 - \
            norm.cdf(pickAfter, player['adp'], player['adp_sd'])

        # For the round after...
        ceiling = roundUp(pickAfter, num_teams)
        leftTillEndOfRound = ceiling - pickAfter
        pickEvenAfter = ceiling + leftTillEndOfRound + 1

        chanceOfStayingOnThreeRounds = 1 - \
            norm.cdf(pickEvenAfter, player['adp'], player['adp_sd'])

        # Creating a dictionary to represent the new row
        created_row = {
            'name': player['name'],
            'pos': position,
            'preds': player['preds'],
            'pct_better': pct_better,
            'ADP': player['adp'],
            'total_pt_gains': total_pt_gains,
            'chanceOfStayingOnBoard': chanceOfStayingOnBoard,
            'chanceOfStayingOnBoardTwoRounds': chanceOfStayingOnBoardTwoRounds,
            'chanceOfStayingOnThreeRounds': chanceOfStayingOnThreeRounds
        }
        created_rows.append(created_row)

    # Convert the created rows to a DataFrame
    createdDataframe = pd.DataFrame(created_rows)

    # Process for the second dataframe
    second_rows = []

    for _, newRow in createdDataframe.iterrows():
        playerPosition = newRow['pos']

        # Filter and sort
        positionallyFiltered = createdDataframe[createdDataframe['pos']
                                                == playerPosition]
        positionallyFiltered = positionallyFiltered.sort_values(
            by='total_pt_gains', ascending=False).head(12).copy()

        # Initialize new columns
        positionallyFiltered['chance_of_best_option'] = 0
        positionallyFiltered['chance_of_best_option_2'] = 0
        positionallyFiltered['chance_of_best_option_3'] = 0

        for j, row in positionallyFiltered.iterrows():
            better_players = positionallyFiltered[positionallyFiltered['total_pt_gains']
                                                  > row['total_pt_gains']]

            # Calculating probabilities for best option next round
            p_noone_better = np.prod(
                1 - better_players['chanceOfStayingOnBoard'])
            p_avail = row['chanceOfStayingOnBoard']
            p_best_option = p_noone_better * p_avail
            positionallyFiltered.at[j, 'chance_of_best_option'] = p_best_option

            # ... for two rounds later
            p_noone_better = np.prod(
                1 - better_players['chanceOfStayingOnBoardTwoRounds'])
            p_avail = row['chanceOfStayingOnBoardTwoRounds']
            p_best_option = p_noone_better * p_avail
            positionallyFiltered.at[j,
                                    'chance_of_best_option_2'] = p_best_option

            # ... and for three rounds later
            p_noone_better = np.prod(
                1 - better_players['chanceOfStayingOnThreeRounds'])
            p_avail = row['chanceOfStayingOnThreeRounds']
            p_best_option = p_noone_better * p_avail
            positionallyFiltered.at[j,
                                    'chance_of_best_option_3'] = p_best_option

        # Calculating average positional values
        nextRoundValue = np.sum(
            positionallyFiltered['chance_of_best_option'] * positionallyFiltered['total_pt_gains'])
        nextRoundValue = 1000 if np.isinf(nextRoundValue) or np.isnan(
            nextRoundValue) else nextRoundValue
        newRow['valueOverNextRound'] = newRow['total_pt_gains'] - nextRoundValue

        valueOverTwoRounds = np.sum(
            positionallyFiltered['chance_of_best_option_2'] * positionallyFiltered['total_pt_gains'])
        valueOverTwoRounds = 1000 if np.isinf(valueOverTwoRounds) or np.isnan(
            valueOverTwoRounds) else valueOverTwoRounds
        newRow['valueOverTwoRounds'] = newRow['total_pt_gains'] - \
            valueOverTwoRounds

        valueOverThreeRounds = np.sum(
            positionallyFiltered['chance_of_best_option_3'] * positionallyFiltered['total_pt_gains'])
        valueOverThreeRounds = 1000 if np.isinf(valueOverThreeRounds) or np.isnan(
            valueOverThreeRounds) else valueOverThreeRounds
        newRow['valueOverThreeRounds'] = newRow['total_pt_gains'] - \
            valueOverThreeRounds

        second_rows.append(newRow)

    secondDataframe = pd.DataFrame(second_rows)
    secondDataframe['total_pt_gains'] = round(secondDataframe['total_pt_gains'], 1)
    secondDataframe['valueOverNextRound'] = round(secondDataframe['valueOverNextRound'], 1)
    secondDataframe['ADP'] = round(secondDataframe['ADP'], 1)

    # add kickers and defense
    teams = {'NO', 'ARI', 'TEN', 'DET', 'WAS', 'BUF', 'ATL', 'NYJ', 'SF', 'TB', 'CLE', 'MIN', 'PHI', 'LV', 'FA', 'HOU', 'CIN', 'MIA', 'LAC', 'NE', 'PIT', 'IND', 'BAL', 'DAL', 'DEN', 'KC', 'NYG', 'SEA', 'GB', 'JAC', 'CHI', 'CAR', 'LAR'}
    # Create a list to hold the new entries
    new_entries = []

    #default player
    default_player = {
        'name': 'Default Player',
        'pos': 'Any',
        'valueOverNextRound': -101,
        'total_pt_gains': -101
    }

    # Loop over each unique team
    for team in teams:
        kicker_entry = {
            'name': f"{team} Kicker",
            'pos': 'K',
            'valueOverNextRound': -100,
            'total_pt_gains': -100,
            # ... other columns with default or null values
        }
        defense_entry = {
            'name': f"{team} Defense",
            'pos': 'DEF',
            'valueOverNextRound': -99,
            'total_pt_gains': -99,
            # ... other columns with default or null values
        }
        new_entries.append(kicker_entry)
        new_entries.append(defense_entry)
    new_entries.append(default_player)

    # Create a new DataFrame with the same columns as `secondDataframe`
    new_df = pd.DataFrame(new_entries, columns=secondDataframe.columns)

    # Concatenate with the existing DataFrame
    final_df = pd.concat([secondDataframe, new_df], ignore_index=True)
    final_df.fillna(0, inplace = True)

    return final_df

# Example usage
draft_id = "1132519955439915008"
user_id = '721855150913814528'
result = draft_optimize(ppr, numberOfTeams, draft_id,user_id)
print(result)