"""Generate deploy csv"""

import pandas as pd
import argparse
import os
import datetime
from collections import namedtuple
import numpy as np
from param_tennis import Param

__author__ = 'Josh Lamstein'


class Transfer:
    def __init__(self, p, csv):
        self.p = p
        self.df = pd.read_csv(csv)
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        self.feats = []
        cols = self.df.columns
        print('cols', cols)
        self.d = {c: [] for c in cols}

    def make_deploy(self, player_pairings, savebool=True):
        for player_pairing in player_pairings:
            player1 = player_pairing.player1
            player2 = player_pairing.player2
            tourney_date = player_pairing.tourney_date
            surface = player_pairing.surface
            best_of = player_pairing.best_of
            tourney_name = player_pairing.tourney_name
            player1_df = self.df[(self.df.player1_name == player1) | (self.df.player2_name == player1)]
            player2_df = self.df[(self.df.player1_name == player2) | (self.df.player2_name == player2)]
            if len(player1_df) == 0:
                print(f'{player1} not found')
            if len(player2_df) == 0:
                print(f'{player2} not found')
            # Todo: decide which features to transfer over
            self.get_transferable_feats(player1_df, player_name=player1, num=1, tourney_date=tourney_date)
            self.get_transferable_feats(player2_df, player_name=player2, num=2, tourney_date=tourney_date)
            self.update_match_info(tourney_date, surface, best_of, tourney_name)
            self.versus(player1, player2)
        length = len(player_pairings)
        for key in self.d.keys():
            if len(self.d[key])==0:
                self.d[key] = [-10] * length
        data = pd.DataFrame(self.d)
        if savebool:
            savepath = os.path.join(self.p.data_dir, 'deploy.csv')
            data.to_csv(savepath)
        return data

    def get_transferable_feats(self, df, player_name, num, tourney_date=None):
        # Sort dataframe by time
        df = df.sort_values('tourney_date', ascending=False)

        row = df.iloc[0]  # take first row of sorted dataframe
        year = int(tourney_date[:4])
        month = int(tourney_date[4:6])
        day = int(tourney_date[6:])

        # check if player_name is for player1 or player2?
        if row.player1_name == player_name:
            tag = 'player1'
        else:
            tag = 'player2'
        # player_id
        player_id = row[f'{tag}_id']
        player_hand = row[f'{tag}_hand']
        player_age = row[f'{tag}_age']
        player_rank = row[f'{tag}_rank']
        player_elo = row[f'{tag}_elo']
        player_ht = row[f'{tag}_ht']
        player_ioc = row[f'{tag}_ioc']
        # check won last game
        game_winner = row['game_winner'] == int(tag[-1])

        time_since_last_game = datetime.datetime(year, month, day) - datetime.datetime(row.year, row.month, row.day)
        days_elapsed = time_since_last_game.total_seconds() // (24 * 60 * 60)
        # calculate streak
        weeks_inactive = days_elapsed // 7

        if game_winner:
            self.d[f'player{num}_winning_streak'].append(row[f'{tag}_winning_streak'] + 1)
            self.d[f'player{num}_losing_streak'].append(0)
        else:
            self.d[f'player{num}_losing_streak'].append(row[f'{tag}_losing_streak'] + 1)
            self.d[f'player{num}_winning_streak'].append(0)

        self.d[f'player{num}_id'].append(player_id)
        self.d[f'player{num}_name'].append(player_name)
        self.d[f'player{num}_hand'].append(player_hand)
        self.d[f'player{num}_age'].append(player_age)
        self.d[f'player{num}_rank'].append(player_rank)
        self.d[f'player{num}_elo'].append(player_elo)
        self.d[f'player{num}_ht'].append(player_ht)
        self.d[f'player{num}_ioc'].append(player_ioc)
        self.d[f'player{num}_weeks_inactive'].append(weeks_inactive)
        #
        checking_two_weeks = weeks_inactive
        game_cnt = 0
        while checking_two_weeks < 2:
            for i, _row in df.iterrows():
                time_since_last_game = datetime.datetime(year, month, day) - datetime.datetime(_row.year, _row.month,
                                                                                               _row.day)
                days_elapsed = time_since_last_game.total_seconds() // (24 * 60 * 60)
                # calculate streak
                checking_two_weeks = days_elapsed // 7
                game_cnt+=1

        self.d[f'player{num}_last_two_weeks'].append(game_cnt)

    def versus(self, player1, player2):
        """Check if players have faced each other"""
        one_wins = 0
        two_wins = 0

        versus12_df = self.df[(self.df.player1_name == player1) & (self.df.player2_name == player2)]
        versus21_df = self.df[(self.df.player1_name == player2) & (self.df.player2_name == player1)]
        if versus12_df is not None:
            one_wins += len(versus12_df[versus12_df.game_winner==1])
            two_wins += len(versus12_df[versus12_df.game_winner==2])
        if versus21_df is not None:
            one_wins += len(versus21_df[versus21_df.game_winner == 1])
            two_wins += len(versus21_df[versus21_df.game_winner == 2])
        self.d['player1_v_player2_wins'].append(one_wins)
        self.d['player2_v_player1_wins'].append(two_wins)

    def update_match_info(self, tourney_date, surface, best_of, tourney_name):
        if tourney_date is not None:
            self.d['tourney_date'].append(tourney_date)
            year = int(tourney_date[:4])
            month = int(tourney_date[4:6])
            day = int(tourney_date[6:])
            self.d['year'].append(year)
            self.d['month'].append(month)
            self.d['day'].append(day)
            yday = datetime.datetime(year=year, month=month, day=day).timetuple().tm_yday
            self.d['yday'].append(yday)
            self.d['cosine_day'].append(np.cos(yday * np.pi / 365))
            self.d['sine_day'].append(np.sin(yday * np.pi / 365))
        if surface is not None:
            self.d['surface'].append(surface)
        if best_of is not None:
            self.d['best_of'].append(best_of)
        if tourney_name is not None:
            self.d['tourney_name'].append(tourney_name)
            tourney_id = self.df.loc[self.df.tourney_name.str.match(tourney_name, case=False), 'tourney_id'].iloc[0]
            self.d['tourney_id'].append(tourney_id)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='/Users/gandalf/Data/tennis/tennis_data/atp_database.csv',
                        help='Input csv generated from clean_data.py for training and analysis.')
    parser.add_argument('--parentdir', default='/Users/gandalf/Data/tennis',
                        help='Parent directory for tennis analysis')
    args = parser.parse_args()
    print(f"Arguments: {args}")
    p = Param(datadir=args.parentdir)
    Player_Pairings = namedtuple('Player_Pairings', 'player1 player2 tourney_date surface best_of tourney_name')
    pairings = []
    pairings.append(Player_Pairings('Jordan Thompson', 'Daniil Medvedev', '20230309', 3, 5, "Indian Wells"))
    pairings.append(Player_Pairings('Altug Celikbilek', 'Vitaliy Sachko', '20230309', 3, 5, "Indian Wells"))
    pairings.append(Player_Pairings('Pablo Andujar', 'Nuno Borges', '20230307', 3, 5, "Indian Wells"))
    pairings.append(Player_Pairings('Leandro Riedi', 'Alexandre Muller', '20230307', 3, 5, "Indian Wells"))
    pairings.append(Player_Pairings('Cristian Garin', 'Filip Misolic', '20230307', 3, 5, "Indian Wells"))
    pairings.append(Player_Pairings('Filip Misolic', 'Cristian Garin', '20230307', 3, 5, "Indian Wells"))
    # player_pairings = [['Jordan Thompson', 'Daniil Medvedev', '20230309', 3],
    #                    ['Altug Celikbilek', 'Vitaliy Sachko', '20230309', 3]]
    Trans = Transfer(p, args.csv)
    Trans.make_deploy(pairings)
