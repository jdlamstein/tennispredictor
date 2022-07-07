import pandas as pd
import os
import glob

def check_player_in_df(csv, player_lst):
    df = pd.read_csv(csv)
    for player_name in player_lst:
        player_df = df[(df.player1_name==player_name) | (df.player2_name==player_name)]
        print(player_df.loc[:, ['player1_name', 'player2_name']])

if __name__=='__main__':
    csv = r'D:\Data\Sports\tennis\tennis_data\atp_database.csv'
    player_name=['Austin Krajicek','Marcel Granollers']
    check_player_in_df(csv, player_name)