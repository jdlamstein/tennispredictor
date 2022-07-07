"""Generate deploy csv"""

import pandas as pd


class Transfer:
    def __init__(self, csv):
        self.df = pd.read_csv(csv)
        self.feats = []

    def make_deploy(self, player_pairings):
        for player1, player2 in player_pairings:
            player1_df = self.df[(self.df.player1_name == player1) | (self.df.player2_name == player1)]
            player2_df = self.df[(self.df.player1_name == player2) | (self.df.player2_name == player2)]
            if len(player1_df) == 0:
                print(f'{player1} not found')
            if len(player2_df) == 0:
                print(f'{player2} not found')
            # Todo: decide which features to transfer over

    def get_transferable_feats(self, df, player_name):
        surface = pd.unique(df.surface)
        # bestof
        #
        cols = df.columns
        df1 = df[df.player1_name == player_name]
        for c in df1.columns:
            if 'player1' in c.lower() and 'name' not in c.lower():
                target = df1[c]
                targ = pd.unique(target)
        df2 = df[df.player2_name == player_name]
        for c in df2.columns:
            if 'player2' in c.lower() and 'name' not in c.lower():
                target = df2[c]
                targ = pd.unique(target)
        df2 = df[df.player2_name == player_name]


if __name__ == '__main__':
    csv = r'D:\Data\Sports\tennis\tennis_data\atp_database.csv'
    player_pairings = [['Austin Krajicek', 'Marcel Granollers'],
                       ['Steve Johnson', 'Horacio Zeballos']]
    make_deploy(csv, player_pairings)
