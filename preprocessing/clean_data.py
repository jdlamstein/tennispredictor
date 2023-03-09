"""From Jeff Sackman's repo, https://github.com/JeffSackmann/tennis_atp,
gather csvs and unite into Pandas dataframe.

The data is arranged by winner and loser. Because we don't know who will win or lose, I set the winner and loser to player1
and player2 and keep the win/lose info in another column."""

import glob
import os
import pandas as pd
import numpy as np
import random
import argparse
from preprocessing.pipeline import Elo, TimePeriod, RecentMatches

__author__='Josh Lamstein'
if os.name == 'nt':
    slash = '\\'
else:
    slash = '/'


class ATP:
    def __init__(self, raw_parent, save_parent):
        # collect csvs of matches
        matches = glob.glob(os.path.join(raw_parent, 'atp_matches_[0-3]*.csv'))
        # collect csvs of futures
        futures = glob.glob(os.path.join(raw_parent, 'atp_matches_futures*.csv'))
        # collect csvs of quals
        quals = glob.glob(os.path.join(raw_parent, 'atp_matches_qual*.csv'))
        self.csvs = matches + futures + quals
        self.savedir = save_parent
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        self.str_labels = ['tourney_id', 'tourney_name']
        self.num_labels = ['winner_id', 'match_id']
        self.name = None
        self.df = None
        self.database = None
        self.savebase = os.path.join(save_parent, 'atp_database.csv')

    def clean_main(self):
        print('Joining and parsing csvs.')
        for csv in self.csvs:
            self.load_df(csv)
            if self.database is None:
                self.database = self.df
            else:
                self.database = pd.concat([self.database, self.df], axis=0, ignore_index=True)
        self.parse_entry()
        self.parse_hand()
        self.parse_round()
        self.parse_ioc()
        self.parse_surface()
        self.parse_tourney_id()
        self.parse_tourney_level()
        self.parse_score()
        self.replace_winner_loser_with_1_2()
        self.scramble_player1_player2_cols()
        self.drop_score()
        self.clean_seed()
        print('Running data checks')
        self.checks(self.database)
        print(f'Saving dataframe to {self.savebase}')
        self.database.to_csv(self.savebase)
        return self.savebase

    def load_df(self, csv):
        self.df = pd.read_csv(csv)
        self.name = csv.split(slash)[-1]

    def drop_l_w(self):
        """drop win/lose columns, already converted them to player1 and player2."""
        self.database = self.database.loc[:, ~self.database.columns.str.startswith('w_')]
        self.database = self.database.loc[:, ~self.database.columns.str.startswith('l_')]

    def set_l_w(self):
        """Find who is the winner and loser, build columns accordingly, replace w_ with player1 and l_ with player2."""
        wcols = self.database.columns[self.database.columns.str.startswith('w_')].to_list()
        lcols = self.database.columns[self.database.columns.str.startswith('l_')].to_list()
        p1_cols = [col.replace('w_', 'player1_') for col in wcols]
        p2_cols = [col.replace('w_', 'player2_') for col in wcols]

        # initialize columns
        for col in p1_cols:
            self.database[col] = -10
        for col in p2_cols:
            self.database[col] = -10

        p1_win_idx = self.database[self.database.game_winner == 1].index
        p2_win_idx = self.database[self.database.game_winner == 2].index
        self.database[p1_cols] = self.database.loc[p1_win_idx, wcols]
        self.database[p2_cols] = self.database.loc[p1_win_idx, lcols]
        self.database[p1_cols] = self.database.loc[p2_win_idx, lcols]
        self.database[p2_cols] = self.database.loc[p2_win_idx, wcols]

    def _replace_uni(self, col, rep=None):
        """
        Convert unique strings to integers
        :param col: column name
        :param rep: rep dict which is returned from this function
        :return: rep dict
        """
        uni = pd.unique(self.database[col].dropna())
        _rep = {col: {}}
        repet = {}
        if rep is None:
            for i, u in enumerate(uni):
                _rep[col][u] = i  # {'a': {'b': np.nan}} in col a replace b with nan
            repet = _rep
        else:
            for key, dct in rep.items():
                repet[col] = dct
        self.database = self.database.replace(to_replace=repet)
        return rep

    def parse_surface(self):
        """Replace unique characters with numeric values."""
        self._replace_uni('surface')

    def parse_tourney_level(self):
        """Replace unique characters with numeric values."""
        self._replace_uni('tourney_level')

    def parse_tourney_id(self):
        """Replace unique characters with numeric values."""
        self.database.tourney_id = self.database.tourney_id.str.split('-').str.join('')

    def parse_hand(self):
        """Replace unique characters with numeric values."""
        rep = self._replace_uni('winner_hand')
        self._replace_uni('loser_hand', rep)

    def clean_seed(self):
        """Convert seed to numeric
        From matches_data_dictionary.txt in the tennis_atp data repo,
        'WC' = wild card, 'Q' = qualifier, 'LL' = lucky loser, 'PR' = protected ranking, 'ITF' = ITF entry,
        and there are a few others that are occasionally used."""
        self.database.player1_seed = self.database.player1_seed.replace({'LL': -1, 'Q': -2, 'WC': -2, 'ITF': -3})
        self.database.player1_seed = self.database.player1_seed.astype(float)
        self.database.player2_seed = self.database.player2_seed.replace({'LL': -1, 'Q': -2, 'WC': -2, 'ITF': -3})
        self.database.player2_seed = self.database.player2_seed.astype(float)

    def parse_ioc(self):
        """Three character country code"""
        rep = self._replace_uni('winner_ioc')
        self._replace_uni('loser_ioc', rep)

    def parse_round(self):
        self._replace_uni('round')

    def parse_entry(self):
        rep = self._replace_uni('winner_entry')
        self._replace_uni('loser_entry', rep)

    def parse_score(self):
        """
        Parse score.
        Scores are written as strings separated by hyphens. The max number of possible games is included in the joined dataframe.
        The columns are populated in a loop.

        Dropping games with exceptions in scoring. Most exceptions are from withdrawals, walkovers, and retirements due to
        illness, injury, or emergency.
        """

        self.database = self.database.drop(self.database[self.database.score.isna()].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('W')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('O')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('/')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('RET')].index)  # retirement
        self.database = self.database.drop(self.database[self.database.score.str.contains('DEF')].index)  # default
        self.database = self.database.drop(self.database[self.database.score.str.contains('ABN')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('UNP')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('NA')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('Default')].index)  # default
        self.database = self.database.drop(self.database[self.database.score.str.contains('Def.')].index)  # default
        self.database = self.database.drop(self.database[self.database.score.str.contains('In')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('Mar')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('Apr')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('Unfinished')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('Feb')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('May')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('Jun')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('Jul')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('RE')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('>')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('Played')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('ABD')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('UNK')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('\?')].index)
        self.database = self.database.drop(self.database[self.database.score.str.contains('nbs')].index)
        game_scores = self.database.score.str.split(' ', n=-1, expand=True)
        player_scores = None
        for col in game_scores.columns:
            player_score = game_scores[col].str.split('-', n=-1, expand=True)
            if len(player_score.columns) == 2:
                player_score.columns = [f'winner_score_{col + 1}', f'loser_score_{col + 1}']
                if player_scores is None:
                    player_scores = player_score
                else:
                    player_scores = pd.concat([player_scores, player_score], axis=1)
        self.database = pd.concat([self.database, player_scores], axis=1)
        for col in self.database.columns:
            if ('winner_score_' in col) or ('loser_score_' in col):
                _new = self.database[col].str.split('(', n=0, expand=True)
                self.database[col] = _new[0]
                _new = self.database[col].str.split('[', n=0, expand=True)
                self.database[col] = _new[0]
                self.database[col] = self.database[col].str.replace(r']', '', regex=True)
                self.database[col] = self.database[col].replace('', np.nan)
                self.database[col] = self.database[col].astype(float)

    def replace_winner_loser_with_1_2(self):
        """Like the function name says, replace the string winner/loser with player1/player2"""
        orig_cols = self.database.columns
        for col in orig_cols:
            newcol = col
            flag = False
            if ('winner' in col):
                newcol = newcol.replace('winner', 'player1')
            elif ('w_' in col) and ('draw' not in col):
                newcol = newcol.replace('w_', 'player1_')
            elif ('loser' in col):
                newcol = newcol.replace('loser', 'player2')
            elif ('l_' in col):
                newcol = newcol.replace('l_', 'player2_')
            if newcol != col:
                self.database[newcol] = self.database[col]
                self.database = self.database.drop(columns=col)
        self.database['game_winner'] = 1

    def scramble_player1_player2_cols(self):
        """
        Player1 was the winner col, player2 is the loser col, must randomize that or the model will know.
        :return:
        """
        cols1 = [c for c in self.database.columns if 'player1' in c]
        cols2 = [c for c in self.database.columns if 'player2' in c]
        cols2_targ = [c.replace('player1', 'player2') for c in cols1]
        assert len(cols2) == len(cols2_targ)
        for c in cols2:
            assert c in cols2_targ
        copydf = self.database.copy()
        maskidx = [i for i in self.database.index if i % 2]
        self.database.loc[maskidx, cols1] = copydf.loc[maskidx, cols2_targ].values
        self.database.loc[maskidx, cols2_targ] = copydf.loc[maskidx, cols1].values
        self.database.loc[maskidx, 'game_winner'] = 2

    def drop_score(self):
        self.database = self.database.drop(columns='score')

    def checks(self, df):
        """Sanity check on players are not playing each other."""
        chk = df.player1_id.equals(df.player2_id)
        assert np.all(chk == False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tennisdir', default='/Users/gandalf/Data/tennis/tennis_atp',
                        help='Data of multiple csvs to be joined for data cleaning.')
    parser.add_argument('--savedir', default='/Users/gandalf/Data/tennis/tennis_data',
                        help='Directory to save joined csv of tennis_atp results.')

    args = parser.parse_args()
    print(f'Arguments: {args}')
    tennis = ATP(args.tennisdir, args.savedir)
    csv = tennis.clean_main()
    print("Calculating ELO score.")
    E = Elo(csv)
    E.populate_elo()
    print("Setting datetime to be sinusoidal.")
    TP = TimePeriod(csv)
    TP.run()
    print("Recording win streak and lose streak over last 2 weeks.")
    Recent = RecentMatches(csv)
    Recent.run()
