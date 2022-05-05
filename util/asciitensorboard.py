#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import numpy as np
import plotext as plt


# custom functions
# -----
from afterburner import read_multiple_runs, convert_em_to_df






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", "-d",  default='./', type=str)
    parser.add_argument("--stream", "-s",  default=False, type=bool)
    parser.add_argument("--rate", "-r", default=30, type=int)

    args = parser.parse_args()
    
    print(args)
    
    # read logdir argument, read the stream argument, read the refreshrate argument
    
    # read the directory
    em = read_multiple_runs(args.logdir)
    df = convert_em_to_df(em)
    # display options of data to display, let user choose with a number
    
    # wait on user input
    #print(df.keys())
    #print(df['training'].keys())
    # display plot, go backplaying of options, or stream continuously
    
    for k1 in df.keys():
        for k2 in df[k1].keys():
            plt.plot(df[k1][k2].values)
            plt.title(f"{k1}_{k2}")
            plt.show()
            plt.clf()
    # plt.plot(df['agent']['agent/sub/actor/weights/min'].values, df['training']['step'].values)
    # plt.title('agent/sub/actor/weights/min')
    #plt.plot(df['agent']['agent/meta/actor/weights/min'].values)

    #plt.show()


# _____________________________________________________________________________


# import the tensorboard reader functions written for saturn or titan?
# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
