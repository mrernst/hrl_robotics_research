#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import numpy as np
import pandas as pd

import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import collections
from matplotlib.patches import Rectangle
from scipy import stats

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_multiplexer

# custom functions
# -----


def read_single_summary(path_to_tfevent, chist=0, img=0, audio=0, scalars=0,
                        hist=0):
    ea = event_accumulator.EventAccumulator(path_to_tfevent, size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: chist,
            event_accumulator.IMAGES: img,
            event_accumulator.AUDIO: audio,
            event_accumulator.SCALARS: scalars,
            event_accumulator.HISTOGRAMS: hist,
                                            })
    ea.Reload()
    ea.Tags()
    return ea


def read_multiple_runs(path_to_project, chist=0, img=0, audio=0, scalars=0,
                       hist=0):
    # use with event_multiplexer (multiplexes different events together
    # useful for retraining I guess...)
    em = event_multiplexer.EventMultiplexer(size_guidance={
        event_accumulator.COMPRESSED_HISTOGRAMS: chist,
        event_accumulator.IMAGES: img,
        event_accumulator.AUDIO: audio,
        event_accumulator.SCALARS: scalars,
        event_accumulator.HISTOGRAMS: hist,
    })
    em.AddRunsFromDirectory(path_to_project)
    # load data
    em.Reload()
    return em


def convert_em_to_df(multiplexer):
    # this needs to be better and be able to cope with different scales
    # sort into training and testing/network
    df_dict = {}
    
    if len(multiplexer.Runs()) == 1:
        # figure out separate runs progressively
        entries = {}
        for run in multiplexer.Runs().keys():
            for tag in multiplexer.Runs()[run]["scalars"]:
                if tag.split('/')[0] not in entries.keys():
                    entries[tag.split('/')[0]] = []
                entries[tag.split('/')[0]].append(tag)
        
        for run in entries:
            run_df = pd.DataFrame()
            for tag in entries[run]:
                tag_df = pd.DataFrame(multiplexer.Scalars(list(multiplexer.Runs().keys())[0], tag))
                tag_df = tag_df.drop(tag_df.columns[[0]], axis=1)
                run_df[tag] = tag_df.value
                run_df["step"] = tag_df.step
            df_dict[run] = run_df

    else:
        for run in multiplexer.Runs().keys():
            # create fresh empty dataframe
            run_df = pd.DataFrame()
            for tag in multiplexer.Runs()[run]["scalars"]:
                tag_df = pd.DataFrame(multiplexer.Scalars(run, tag))
                tag_df = tag_df.drop(tag_df.columns[[0]], axis=1)
                run_df[tag] = tag_df.value
                run_df["step"] = tag_df.step
            df_dict[run] = run_df

    return df_dict


# ----------------
# Afterburner Classes
# ----------------


class DataEssence(object):
    """docstring for DataEssence."""

    def __init__(self):
        super(DataEssence, self).__init__()

    def write_to_file(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.essence, output, pickle.HIGHEST_PROTOCOL)

    def read_from_file(self, filename):
        with open(filename, 'rb') as input:
            self.essence = pickle.load(input)

    def distill(self, path, evaluation_data, embedding_data=None):
        self.essence = self._read_tfevents(path)

        # needless_keys = [l for l in list(self.essence.keys()) if '/' in l]
        # for key in needless_keys:
        # 	self.essence.pop(key, None)

        self.essence['evaluation'] = evaluation_data
        if embedding_data:
            self.essence['embedding'] = embedding_data

    def _read_tfevents(self, path):
        em = read_multiple_runs(path)
        df = convert_em_to_df(em)
        return df

    def plot_essentials(self, savefile):
        # start figure
        fig, axes = plt.subplots(3, 3, sharex='all', figsize=[7, 7])
        # plot images onto figure
        self._plot_traintest_lcurve(axes)
        self._plot_parameter_lcurve(axes)

        # save figure
        fig.suptitle(savefile.rsplit('/')[-1])
        fig.savefig(savefile)
        pass
    
    # TODO: write visualizer functions
    def _plot_traintest_lcurve(self, axes):
        pass

    def _plot_timebased_lcurve(self, axes):
        pass

    def _plot_parameter_lcurve(self, axes):
        pass


class EssenceCollection(object):
    """docstring for EssenceCollection."""

    def __init__(self, remove_files=False):
        super(EssenceCollection, self).__init__()
        path_to_file = os.path.realpath(__file__)
        self.path_to_experiment = path_to_file.rsplit('/', 3)[0]
        self.collection = self.collect_data_essences(
            self.path_to_experiment, remove_files)

    def collect_data_essences(self, path_to_experiment, remove_files):
        # gather and read all files in files/essence/
        collection = {}
        essence = DataEssence()
        path_to_data = path_to_experiment + "/data/"
        for file in os.listdir(path_to_data):
            if file.endswith(".pkl") and file.startswith("conf"):
                config_name = file.split('.')[0].rsplit('i', 1)[0]
                iteration_number = file.split('.')[0].rsplit('i', 1)[-1]
                essence.read_from_file(os.path.join(path_to_data, file))
                if config_name in collection.keys():
                    collection[config_name][iteration_number] = essence.essence
                else:
                    collection[config_name] = {}
                    collection[config_name][iteration_number] = essence.essence
                # delete file
                if remove_files:
                    os.remove(os.path.join(path_to_data, file))
                    print("[INFO] File '{}' deleted.".format(file))
                else:
                    pass
        return collection

    def write_to_file(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.collection, output, pickle.HIGHEST_PROTOCOL)

    def read_from_file(self, filename):
        with open(filename, 'rb') as input:
            self.collection = pickle.load(input)






if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--config", type=str)
    # 
    # args = parser.parse_args()
    
    print('[INFO] afterburner running, collecting data')
    ess_coll = EssenceCollection(remove_files=False) # TODO: Change back to True once this works reliably
    
    ess_coll.write_to_file(
        ess_coll.path_to_experiment +
        '/data/{}.pkl'.format(
            ess_coll.path_to_experiment.rsplit('/')[-1]))
    

# _____________________________________________________________________________

if __name__ == "__main__":
    pass

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
