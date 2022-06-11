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

def let_user_pick(options):
    print("Please choose:")
    for idx, element in enumerate(options):
        print("{}) {}".format(idx+1,element))
    i = input("Enter number: ")
    try:
        if 0 < int(i) <= len(options):
            return int(i) - 1
    except:
        pass
    return None




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", "-d",  default='./save/', type=str)
    parser.add_argument("--rate", "-r", default=30, type=int)
    
    parser.add_argument('--present_options', '-o',
                        dest='present_options',
                        action='store_true')
    parser.add_argument('--no-present_options', '-no',
                        dest='present_options',
                        action='store_false')
    parser.set_defaults(present_options=True)
    
    parser.add_argument('--combine_views', '-c',
                        dest='combine_views',
                        action='store_true')
    parser.add_argument('--no-combine_views', '-nc',
                        dest='combine_views',
                        action='store_false')
    parser.set_defaults(combine_views=False)
    
    parser.add_argument('--stream', '-s',
                        dest='stream',
                        action='store_true')
    parser.add_argument('--no-stream', '-ns',
                        dest='stream',
                        action='store_false')
    parser.set_defaults(stream=False)


    
    # read logdir argument, read the stream argument, read the refreshrate argument
    args = parser.parse_args()    
    
    # read the directory
    em = read_multiple_runs(args.logdir)
    df = convert_em_to_df(em)
    
    # display options of data to display, let user choose with a number
    user_picked_run = None
    if args.present_options or args.stream:
        # wait on user input
        user_picked_run = let_user_pick(list(df.keys())+['combine views'])
    #print(df.keys())
    #print(df['training'].keys())
    # display plot, go backplaying of options, or stream continuously
    

    
    if args.combine_views or user_picked_run==len(df.keys()):
        reverse_dict_of_keys = {}
        for k1 in df.keys():
            for k2 in df[k1].keys():
                if k2 in reverse_dict_of_keys.keys():
                    reverse_dict_of_keys[k2].append(k1)
                else:
                    reverse_dict_of_keys[k2] = []
                    reverse_dict_of_keys[k2].append(k1)
        
        if args.stream:
            # show options and ask user
            user_picked_metric = let_user_pick([k for k in reverse_dict_of_keys.keys() if len(reverse_dict_of_keys[k])>1])
            # enter streaming loop
            k2 = list(df[k1].keys())[user_picked_metric]
            plt.title(f"Streaming: {k2}")
            while True:
                plt.clt()
                plt.cld()
                for i, k1 in enumerate(reverse_dict_of_keys[k2]):
                    sets_of_k1 = [set(k1.split('/')) for k1 in reverse_dict_of_keys[k2]]
                    sets_of_k1_without_i = sets_of_k1.copy()
                    _ = sets_of_k1_without_i.pop(i)
                    plt.plot(df[k1][k2].values, label='/'.join(sets_of_k1[i].difference(*sets_of_k1_without_i)))
                plt.show()
                
                plt.sleep(args.rate)
                em.Reload()
                df = convert_em_to_df(em)

        else:
            #plot all
            for k2 in reverse_dict_of_keys.keys():
                if len(reverse_dict_of_keys[k2]) > 1 and ('loss' in k2 or 'reward' in k2 or 'success' in k2):
                    for i, k1 in enumerate(reverse_dict_of_keys[k2]):
                        sets_of_k1 = [set(k1.split('/')) for k1 in reverse_dict_of_keys[k2]]
                        sets_of_k1_without_i = sets_of_k1.copy()
                        _ = sets_of_k1_without_i.pop(i)
                        plt.plot(df[k1][k2].values, label='/'.join(sets_of_k1[i].difference(*sets_of_k1_without_i)))
                    plt.title(f"{k2}")
                    plt.show()
                    plt.clf()
    else:
        if isinstance(user_picked_run, int):
            k1 = list(df.keys())[user_picked_run]
            if args.stream:
                # show options and ask user
                user_picked_metric = let_user_pick(list(df[k1].keys()))
                # enter streaming loop
                k2 = list(df[k1].keys())[user_picked_metric]
                plt.title(f"Streaming: {k2}")
                while True:
                    plt.clt()
                    plt.cld()
                    plt.plot(df[k1][k2].values)
                    plt.show()
                    
                    plt.sleep(args.rate)
                    em.Reload()
                    df = convert_em_to_df(em)
            else:
                #plot all
                for k2 in df[k1].keys():            
                    if 'loss' in k2 or 'reward' in k2 or 'success' in k2:
                        plt.plot(df[k1][k2].values)
                        plt.title(f"{k2}")
                        plt.show()
                        plt.clf()
        else:
            # plot everything
            for k1 in df.keys():
                for k2 in df[k1].keys():            
                    if 'loss' in k2 or 'reward' in k2 or 'success' in k2:
                        plt.plot(df[k1][k2].values)
                        plt.title(f"{k2}")
                        plt.show()
                        plt.clf()



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
