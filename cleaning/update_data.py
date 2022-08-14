"""Get updates from github, and recreate data csv."""
import shutil

from preprocessing.pipeline import Elo, TimePeriod, RecentMatches
from cleaning.clean_data import ATP


def backup(csv, step):
    parts = csv.split('\\')
    name = parts[-1].split('.c')[0]
    name += f'_{step}.csv'
    parts = parts[:-1] + [name]
    savename = '\\'.join(parts)

    shutil.copy2(csv, savename)


if __name__ == '__main__':
    raw_parent = r'D:\Data\Sports\tennis\tennis_atp'
    save_parent = r'D:\Data\Sports\tennis\tennis_data'
    tennis = ATP(raw_parent, save_parent)
    csv = tennis.clean_main()
    backup(csv, 'atp')
    E = Elo(csv)
    E.populate_elo()
    backup(csv, 'elo')

    TP = TimePeriod(csv)
    TP.run()
    backup(csv, 'timeperiod')
    Recent = RecentMatches(csv)
    Recent.run()
    backup(csv, 'matches')
