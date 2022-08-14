import pandas as pd
import param_tennis as param
import os
import numpy as np
import datetime
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F


class Elo:
    def __init__(self, csv):
        self.csv = csv
        self.df = pd.read_csv(self.csv)
        self.elo = {}  # id key, list of elos
        self.df = self.df.sort_values(by=['tourney_date', 'round'])
        self.df.player1_elo = 1500
        self.df.player2_elo = 1500
        self.res = self.df.copy()

    def populate_elo(self):
        # chronologically
        # init rating as 1500
        # compete as a tree,
        # update elo for next match
        # store elo in dict per player
        # update per row
        for i, row in self.df.iterrows():

            id1 = row.player1_id
            id2 = row.player2_id
            winner = row.game_winner  # 1 or 2
            if id1 not in self.elo.keys():
                self.elo[id1] = [1500]
            if id2 not in self.elo.keys():
                self.elo[id2] = [1500]
            elo_1 = self.calc_elo(self.elo[id1][-1], self.elo[id2][-1], winner % 2, len(self.elo[id1]))
            elo_2 = self.calc_elo(self.elo[id2][-1], self.elo[id1][-1], winner - 1, len(self.elo[id2]))
            self.elo[id1].append(elo_1)
            self.elo[id2].append(elo_2)
            self.res.loc[i, ['player1_elo', 'player2_elo']] = [elo_1, elo_2]
        self.res.to_csv(self.csv)
        return self.csv

    def calc_elo(self, rating_a, rating_b, winner, num_a_games):
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        if num_a_games < 30 and rating_a < 2400:
            K = 40
        elif rating_a < 2400:
            K = 20
        else:
            K = 10
        rating_a_prime = rating_a + K * (winner - expected_a)
        return rating_a_prime


class Glicko:
    def __init__(self, csv):
        self.csv = csv
        self.df = pd.read_csv(self.csv)
        self.rd = {}  # id key, list of elos
        self.sigma = {}  # id key, list of elos
        self.df = self.df.sort_values(by=['tourney_date', 'round'])
        self.df.player1_rd = 350
        self.df.player2_rd = 350
        self.res = self.df.copy()

    def populate_elo(self):
        # chronologically
        # init rating as 1500
        # compete as a tree,
        # update elo for next match
        # store elo in dict per player
        # update per row
        for i, row in self.df.iterrows():

            id1 = row.player1_id
            id2 = row.player2_id
            winner = row.game_winner  # 1 or 2
            if id1 not in self.elo.keys():
                self.elo[id1] = [1500]
            if id2 not in self.elo.keys():
                self.elo[id2] = [1500]
            elo_1 = self.calc_elo(self.elo[id1][-1], self.elo[id2][-1], winner % 2, len(self.elo[id1]))
            elo_2 = self.calc_elo(self.elo[id2][-1], self.elo[id1][-1], winner - 1, len(self.elo[id2]))
            self.elo[id1].append(elo_1)
            self.elo[id2].append(elo_2)
            self.res.loc[i, ['player1_elo', 'player2_elo']] = [elo_1, elo_2]
        self.res.to_csv(self.csv)

    def calc_elo(self, rating_a, rating_b, winner, num_a_games):
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        if num_a_games < 30 and rating_a < 2400:
            K = 40
        elif rating_a < 2400:
            K = 20
        else:
            K = 10
        rating_a_prime = rating_a + K * (winner - expected_a)
        return rating_a_prime


class TimePeriod:
    # Set year and enter time as sin(x* 2pi/365) cos(x*2pi/365)
    def __init__(self, csv):
        self.csv = csv
        self.df = pd.read_csv(self.csv)
        self.df = self.df.sort_values(by=['tourney_date', 'round'])

    def run(self):
        self.get_year()
        self.get_day()
        self.df.to_csv(self.csv, index=False)
        return self.csv

    def get_year(self):
        self.df['year'] = self.df.tourney_date // 10000

    def get_day(self):
        self.df['sine_day'] = -2
        self.df['cosine_day'] = -2
        monthday = self.df.tourney_date % 10000
        self.df['month'] = monthday // 100
        self.df['day'] = monthday % 100
        yday = self.df.apply(lambda x: datetime.datetime(year=x.year, month=x.month, day=x.day).timetuple().tm_yday,
                             axis=1)
        # for i, row in tqdm(self.df.iterrows()):
        #     monthday = row.tourney_date % 10000
        #     month = monthday // 100
        #     day = monthday % 100
        #     yday = datetime.datetime(year=row.year, month=month, day=day).timetuple().tm_yday
        self.df['yday'] = yday
        self.df['sine_day'] = np.sin(yday * np.pi / 365)
        self.df['cosine_day'] = np.cos(yday * np.pi / 365)


class Dataspring:
    def __init__(self, csv):
        self.p = param.Param()
        self.csv = csv
        self.name = csv.split('\\')[-1].split('.')[0]
        self.miss_lst = []
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None
        self.columns = None

    def remove_cols_with_name_in_them(self, df):
        namelst = []
        for col in df.columns:
            if 'name' in col:
                namelst.append(col)
        df = df.drop(columns=namelst)
        return df

    def get_col_types(self, df):
        types = []
        for col in df.columns:
            uni = pd.unique(df[col].dropna())
            typ_lst = []
            for u in uni:
                typ_lst.append(type(u).__name__)
            typ_uni = np.unique(typ_lst)
            if len(typ_uni) > 1:
                # print(col)
                # print(typ_uni)
                types.append('str')
            elif len(typ_uni) == 0:
                print('NO UNIQUES')
                print(col)
                print(uni)
                print(type(u))
                types.append('float')
            else:
                if typ_uni[0] == 'str':
                    print(col)
                    print(typ_uni)
                types.append(typ_uni[0])
        final_uni, cnt_uni = final_types = np.unique(types, return_counts=True)
        print(final_uni)
        print(cnt_uni)
        return types

    # def csv_to_ds(self):
    #     df = pd.read_csv(self.csv)
    #     df_train, df_val, df_test, self.labels_train, self.labels_val, self.labels_test = self.process_df(df)
    #     df_train.to_csv(self.csv_train)
    #     df_val.to_csv(self.csv_val)
    #     df_test.to_csv(self.csv_test)
    #
    #     self.ds_train = tf.data.experimental.make_csv_dataset(self.csv_train, batch_size=self.p.batch_size,
    #                                                           column_names=df.columns)
    #     self.ds_val = tf.data.experimental.make_csv_dataset(self.csv_val, batch_size=self.p.batch_size,
    #                                                         column_names=df.columns)
    #     self.ds_test = tf.data.experimental.make_csv_dataset(self.csv_test, batch_size=self.p.batch_size,
    #                                                          column_names=df.columns)
    #
    # def dict_to_ds_with_labels(self):
    #     df = pd.read_csv(self.csv)
    #
    #     df = self.remove_cols_with_name_in_them(df)
    #
    #     df_train, df_val, df_test, self.labels_train, self.labels_val, self.labels_test = self.process_df(df)
    #     print('train columns', pd.unique(df_train.columns))
    #
    #     df_train, mean, std = self.derive_and_norm(df_train)
    #     df_val = self.norm_df(df_val, mean, std)
    #     df_test = self.norm_df(df_test, mean, std)
    #     # todo: save mean and std for deployment, log in wandb
    #     feats_train = {name: np.array(value)
    #                    for name, value in df_train.items()}
    #     feats_val = {name: np.array(value)
    #                  for name, value in df_val.items()}
    #     feats_test = {name: np.array(value)
    #                   for name, value in df_test.items()}
    #     print('feature length', len(feats_train))
    #
    #     if self.p.output_size==2:
    #         self.labels_train = tf.one_hot(self.labels_train, 2)
    #         self.labels_val = tf.one_hot(self.labels_val, 2)
    #         self.labels_test = tf.one_hot(self.labels_test, 2)
    #     self.ds_train = tf.data.Dataset.from_tensor_slices((feats_train, self.labels_train))
    #     self.ds_val = tf.data.Dataset.from_tensor_slices((feats_val, self.labels_val))
    #     self.ds_test = tf.data.Dataset.from_tensor_slices((feats_test, self.labels_test))
    #     return feats_train, feats_val, feats_test

    def build_dataset_with_labels(self):
        feats_train, feats_val, feats_test, labels_train, labels_val, labels_test = self.prepare_dataset()

        dataset_train = TensorDataset(torch.Tensor(feats_train),
                                      F.one_hot(torch.Tensor(labels_train).to(torch.int64), self.p.output_size))
        dataset_val = TensorDataset(torch.Tensor(feats_val),
                                    F.one_hot(torch.Tensor(labels_val).to(torch.int64), self.p.output_size))
        dataset_test = TensorDataset(torch.Tensor(feats_test),
                                     F.one_hot(torch.Tensor(labels_test).to(torch.int64), self.p.output_size))
        return dataset_train, dataset_val, dataset_test

    def prepare_dataset(self):
        df = pd.read_csv(self.csv)

        df = self.remove_cols_with_name_in_them(df)

        df_train, df_val, df_test, labels_train, labels_val, labels_test = self.process_df(df)
        print('train columns', pd.unique(df_train.columns))

        df_train, mean, std = self.derive_and_norm(df_train)
        df_val = self.norm_df(df_val, mean, std)
        df_test = self.norm_df(df_test, mean, std)
        # todo: save mean and std for deployment, log in wandb
        feats_train = df_train.to_numpy()
        feats_val = df_val.to_numpy()
        feats_test = df_test.to_numpy()
        print('feature length', len(feats_train))
        return feats_train, feats_val, feats_test, labels_train, labels_val, labels_test

    # def dict_to_ds_deploy(self):
    #     df = pd.read_csv(self.csv)
    #     df_deploy, labels_deploy = self.process_df_deploy(df)
    #     df_deploy, mean, std = self.derive_and_norm(df_deploy)
    #     feats_deploy = {name: np.array(value)
    #                     for name, value in df_deploy.items()}
    #     print('feature length', len(feats_deploy))
    #     self.ds_train = tf.data.Dataset.from_tensor_slices((feats_deploy, labels_deploy))
    #     return feats_deploy, labels_deploy

    def pandas_to_ds(self):
        df = pd.read_csv(self.csv)
        df_train, df_val, df_test, self.labels_train, self.labels_val, self.labels_test = self.process_df(df)

        self.ds_train = tf.data.Dataset.from_tensor_slices((df_train.values, self.labels_train.values))
        self.ds_val = tf.data.Dataset.from_tensor_slices((df_val.values, self.labels_val.values))
        self.ds_test = tf.data.Dataset.from_tensor_slices((df_test.values, self.labels_test.values))
        return None, None, None

    def datagen_base(self):
        self.ds_train = self.ds_train.shuffle(1000).batch(self.p.batch_size)
        self.ds_val = self.ds_val.shuffle(1000).batch(self.p.batch_size)
        self.ds_test = self.ds_test.shuffle(1000).batch(self.p.batch_size)
        self.ds_train.repeat(None)
        self.ds_val.repeat(1)
        self.ds_test.repeat(1)

    def generator(self):
        for row in self.ds:
            yield row

    def process_df(self, df):
        """
        Get labels (game winner atm). Remove unnecessary columns.
        Todo: Later include betting odds
        :param df: data
        :return:
        """
        df = df.sort_values(by=['tourney_date', 'round'])
        df = df.fillna(-10)  # fill nan
        print('shape', df.shape)
        cols = df.columns
        drop_cols = ['player1_score', 'player2_score', 'player1_rank', 'player2_rank']
        for col in cols:
            for drop_col in drop_cols:
                if drop_col in col:
                    df = df.drop(columns=[col])
        drop_label_cols = ['_ace', '_df',
                           '_svpt', '_1stIn',
                           '_1stWon', '_2ndWon',
                           '_SvGms', '_bpSaved', '_bpFaced']
        for col in cols:
            for drop_label_col in drop_label_cols:
                if drop_label_col in col:
                    df = df.drop(columns=[col])
        df = df.drop(columns=['tourney_id'])
        df = df.drop(columns=['month'])
        df = df.drop(columns=['day'])
        df = df.drop(columns=['minutes'])
        df = df.drop(columns=['tourney_date'])
        types = self.get_col_types(df)
        df_train = df.iloc[0:int(len(df) * .6)]
        df_val = df.iloc[int(len(df) * .6):int(len(df) * .8)]
        df_test = df.iloc[int(len(df) * .8):]
        print('len train', len(df_train))
        print('len val', len(df_val))
        print('len test', len(df_test))
        labels_train = np.array(df_train.pop('game_winner') - 1)
        labels_val = np.array(df_val.pop('game_winner') - 1)
        labels_test = np.array(df_test.pop('game_winner') - 1)



        self.columns = df_train.columns
        return df_train, df_val, df_test, labels_train, labels_val, labels_test


    def process_df_deploy(self, df):
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.fillna(-10)  # fill nan
        print('shape', df.shape)
        cols = df.columns

        for col in cols:
            if 'player1_score' in col or 'player2_score' in col or 'Unnamed' in col:
                df = df.drop(columns=[col])
        df = df.drop(columns=['tourney_id'])
        types = self.get_col_types(df)
        assert len(types) == len(df.columns), f'{len(types)} {len(df.columns)}'
        labels = np.ones((len(df),))
        return df, labels


    @staticmethod
    def derive_and_norm(df):
        mean = df.mean(axis=0, skipna=True)
        std = df.std(axis=0, skipna=True)
        res = (df - mean) / std
        return res, mean, std


    @staticmethod
    def norm_df(df, mean, std):
        res = (df - mean) / std
        return res


class RecentMatches:
    """
    Get win streak, lose streak, number of games in last 2 weeks, last game played
    """

    def __init__(self, csv):
        self.csv = csv
        self.df = pd.read_csv(self.csv)

        self.df = self.df.sort_values(by=['tourney_date', 'round'])
        self.res = self.df.copy()
        self.res['player1_winning_streak'] = 0
        self.res['player2_winning_streak'] = 0
        self.res['player1_losing_streak'] = 0
        self.res['player2_losing_streak'] = 0
        self.res['player1_weeks_inactive'] = 0
        self.res['player2_weeks_inactive'] = 0
        self.res['player1_last_two_weeks'] = 0
        self.res['player2_last_two_weeks'] = 0
        self.res['player1_v_player2_wins'] = 0
        self.res['player2_v_player1_wins'] = 0
        self.win_dict = {}
        # make a df of players vs players, winners vs losers
        unique_players = pd.unique(pd.concat((self.df.player1_id, self.df.player2_id)))
        unique_players.sort()
        self.win_df = pd.DataFrame(data=0, columns=unique_players, index=unique_players)
        self.players = {}

    def run(self):
        self.update()
        self.res.to_csv(self.csv, index=False)
        print('Ran recent matches')

    def update(self):
        # get number of games in last two weeks
        # get consecutive wins and losses
        # get weeks inactive
        for i, row in self.df.iterrows():

            id1 = row.player1_id
            id2 = row.player2_id
            winner = row.game_winner
            self.res.loc[i, 'player1_v_player2_wins'] = self.win_df.loc[id1, id2]
            self.res.loc[i, 'player2_v_player1_wins'] = self.win_df.loc[id2, id1]
            if not id1 in self.players.keys():
                self.players[id1] = []
            if not id2 in self.players.keys():
                self.players[id2] = []
            # index winners, columns losers
            if winner == 1:
                self.win_df.loc[id1, id2] += 1
            else:
                self.win_df.loc[id2, id1] += 1
            year = row.year
            day = row.yday
            self.player_util_func(i, id1, year, day, winner, 'player1')
            self.player_util_func(i, id2, year, day, winner, 'player2')

    def player_util_func(self, i, ident, year, day, winner, player_string):
        assert player_string == 'player1' or player_string == 'player2', 'check player string'
        # initialize for when player is introduced
        prev_year = year
        prev_day = day
        while len(self.players[ident]) > 0:
            time_tuple = self.players[ident][0][0]
            prev_year = time_tuple[0]
            prev_day = time_tuple[1]
            if (year == prev_year and day - prev_day < 14) or \
                    (year - prev_year == 1 and 14 > day - prev_day - 365 > 0):
                # within two week range
                break
            else:
                self.players[ident].pop(0)  # remove game from list, outside of 2 week window
        dyear = year - prev_year
        dday = day - (prev_day - dyear * 365)
        weeks = dday // 14
        # weeks inactive
        self.res.loc[i, f'{player_string}_weeks_inactive'] = weeks
        self.players[ident].append([(year, day), winner])
        # games in last two weeks
        self.res.loc[i, f'{player_string}_last_two_weeks'] = len(self.players[ident])

        # check streak within two weeks
        for j, (_, wins) in enumerate(reversed(self.players[ident])):
            if j == 0:
                orig_win = wins
            else:
                if wins != orig_win:
                    if orig_win == 1:
                        self.res.loc[i, f'{player_string}_winning_streak'] = j
                    elif orig_win == 2:
                        self.res.loc[i, f'{player_string}_losing_streak'] = j  # looping through player1 games


if __name__ == '__main__':
    csv = r'D:\Data\Sports\tennis\tennis_data\atp_database.csv'
    Dat = Dataspring(csv)
    dataset_train, dataset_val, dataset_test = Dat.build_dataset_with_labels()


    # E = Elo(csv)
    # E.populate_elo()

    # TP = TimePeriod(csv)
    # TP.run()

    # Recent = RecentMatches(csv)
    # Recent.run()
