"""Clean data, figure out what columns are.
11/26/2020 expecting using atp data"""

import glob
import os
import pandas as pd
import numpy as np
import random


class ATP:
    def __init__(self, raw_parent, save_parent):
        matches = glob.glob(os.path.join(raw_parent, 'atp_matches_[0-3]*.csv'))
        futures = glob.glob(os.path.join(raw_parent, 'atp_matches_futures*.csv'))
        quals = glob.glob(os.path.join(raw_parent, 'atp_matches_qual*.csv'))
        self.csvs = matches + futures+quals
        self.savedir = save_parent

        self.str_labels = ['tourney_id', 'tourney_name']
        self.num_labels = ['winner_id', 'match_id']
        self.name = None
        self.df = None
        self.database = None
        self.savebase = os.path.join(save_parent, 'atp_database.csv')

    def clean_main(self):
        for csv in self.csvs:
            self.load_df(csv)
            # self.drop_l_w()
            self.parse_entry()
            self.parse_hand()
            self.parse_round()
            self.parse_ioc()
            self.parse_surface()
            self.parse_tourney_id()
            self.parse_tourney_level()
            self.parse_score()
            # self.remove_cols_with_name_in_them()
            self.replace_winner_loser_with_1_2()
            self.scramble_player1_player2_cols()
            self.drop_score()
            if self.database is None:
                self.database = self.df
            else:
                self.database = pd.concat([self.database, self.df], axis=0, ignore_index=True)
        self.clean_seed()
        self.checks(self.database)
        self.database.to_csv(self.savebase)
        return self.savebase

    def load_df(self, csv):
        self.df = pd.read_csv(csv)
        self.name = csv.split('\\')[-1]

    def drop_l_w(self):
        self.df = self.df.loc[:, ~self.df.columns.str.startswith('w_')]
        self.df = self.df.loc[:, ~self.df.columns.str.startswith('l_')]

    def set_l_w(self):
        """Find who is the winner and loser, build columns accordingly"""
        wcols = self.df.columns[self.df.columns.str.startswith('w_')].to_list()
        lcols = self.df.columns[self.df.columns.str.startswith('l_')].to_list()
        p1_cols = [col.replace('w_', 'player1_') for col in wcols]
        p2_cols = [col.replace('w_', 'player2_') for col in wcols]

        # initialize columns
        for col in p1_cols:
            self.df[col] = -10
        for col in p2_cols:
            self.df[col] = -10

        p1_win_idx = self.df[self.df.game_winner == 1].index
        p2_win_idx = self.df[self.df.game_winner == 2].index
        self.df[p1_cols] = self.df.loc[p1_win_idx, wcols]
        self.df[p2_cols] = self.df.loc[p1_win_idx, lcols]
        self.df[p1_cols] = self.df.loc[p2_win_idx, lcols]
        self.df[p2_cols] = self.df.loc[p2_win_idx, wcols]

    def _replace_uni(self, col, rep=None):
        """
        Convert unique strings to integers
        :param col: column name
        :param rep: rep dict which is returned from this function
        :return: rep dict
        """
        uni = pd.unique(self.df[col].dropna())
        _rep = {col: {}}
        repet = {}
        if rep is None:
            for i, u in enumerate(uni):
                _rep[col][u] = i  # {'a': {'b': np.nan}} in col a replace b with nan
            repet = _rep
        else:
            for key, dct in rep.items():
                repet[col] = dct
        self.df = self.df.replace(to_replace=repet)
        return rep

    def parse_surface(self):
        self._replace_uni('surface')

    def parse_tourney_level(self):
        self._replace_uni('tourney_level')

    def parse_tourney_id(self):
        self.df.tourney_id = self.df.tourney_id.str.split('-').str.join('')

    def parse_hand(self):
        rep = self._replace_uni('winner_hand')
        self._replace_uni('loser_hand', rep)

    def clean_seed(self):
        self.database.player1_seed = self.database.player1_seed.replace({'LL': -1, 'Q': -2, 'WC': -2})
        self.database.player1_seed = self.database.player1_seed.astype(float)
        self.database.player2_seed = self.database.player2_seed.replace({'LL': -1, 'Q': -2, 'WC': -2})
        self.database.player2_seed = self.database.player2_seed.astype(float)

    # def parse_name(self):
    #     rep = self._replace_uni('winner_name')
    #     self._replace_uni('loser_name', rep)

    def parse_ioc(self):
        rep = self._replace_uni('winner_ioc')
        self._replace_uni('loser_ioc', rep)

    def parse_round(self):
        self._replace_uni('round')

    def parse_entry(self):
        rep = self._replace_uni('winner_entry')
        self._replace_uni('loser_entry', rep)

    def parse_score(self):
        """
        parse score
        :param df:
        :return:
        """

        # df.score
        # W/O, and RET
        # 2 sets vs 3 sets
        # clean tiebreaks

        def assign_score(df, col):
            new = self.df[col].split('-', n=1, expand=True)
            win_col = 'winner_score_' + col.split('_')[-1]
            lose_col = 'loser_score_' + col.split('_')[-1]
            df[win_col] = new[0]
            df[lose_col] = new[1]
            return df

        self.df = self.df.drop(self.df[self.df.score.isna()].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('W')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('O')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('/')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('RET')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('DEF')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('ABN')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('UNP')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('NA')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('Default')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('Def.')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('In')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('Mar')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('Apr')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('Unfinished')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('Feb')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('May')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('Jun')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('Jul')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('RE')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('>')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('Played')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('ABD')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('UNK')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('\?')].index)
        self.df = self.df.drop(self.df[self.df.score.str.contains('nbs')].index)
        game_scores = self.df.score.str.split(' ', n=-1, expand=True)
        player_scores = None
        for col in game_scores.columns:
            player_score = game_scores[col].str.split('-', n=-1, expand=True)
            if len(player_score.columns) == 2:
                player_score.columns = [f'winner_score_{col + 1}', f'loser_score_{col + 1}']
                if player_scores is None:
                    player_scores = player_score
                else:
                    player_scores = pd.concat([player_scores, player_score], axis=1)
        self.df = pd.concat([self.df, player_scores], axis=1)
        for col in self.df.columns:
            if ('winner_score_' in col) or ('loser_score_' in col):
                print(col)
                _new = self.df[col].str.split('(', n=0, expand=True)
                self.df[col] = _new[0]
                _new = self.df[col].str.split('[', n=0, expand=True)
                self.df[col] = _new[0]
                self.df[col] = self.df[col].str.replace(']', '')
                self.df[col] = self.df[col].replace('', np.nan)
                print(pd.unique(self.df[col]))

                self.df[col] = self.df[col].astype(float)
        print(self.df.winner_score_1.dtypes)

    def replace_winner_loser_with_1_2(self):
        orig_cols = self.df.columns
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
                self.df[newcol] = self.df[col]
                self.df = self.df.drop(columns=col)
        self.df['game_winner'] = 1

    def scramble_player1_player2_cols(self):
        """
        Player1 was the winner col, player2 is the loser col, must scramble that.
        :return:
        """
        cols1 = [c for c in self.df.columns if 'player1' in c]
        cols2 = [c for c in self.df.columns if 'player2' in c]
        cols2_targ = [c.replace('player1', 'player2') for c in cols1]
        assert len(cols2) == len(cols2_targ)
        for c in cols2:
            assert c in cols2_targ
        copydf = self.df.copy()
        maskidx = [i for i in self.df.index if i % 2]
        self.df.loc[maskidx, cols1] = copydf.loc[maskidx, cols2_targ].values
        self.df.loc[maskidx, cols2_targ] = copydf.loc[maskidx, cols1].values
        self.df.loc[maskidx, 'game_winner'] = 2

    def drop_score(self):
        self.df = self.df.drop(columns='score')

    def checks(self, df):
        chk = df.player1_id.equals(df.player2_id)
        assert np.all(chk == False)


if __name__ == '__main__':
    raw_parent = r'D:\Data\Sports\tennis\tennis_atp'
    save_parent = r'D:\Data\Sports\tennis\tennis_data'
    tennis = ATP(raw_parent, save_parent)
    tennis.clean_main()
