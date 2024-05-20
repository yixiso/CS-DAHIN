import pathlib

import numpy as np
import scipy.sparse
import scipy.io
from scipy import sparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import utils.preprocess
from sklearn.model_selection import train_test_split
import pickle as pkl
import os



def pre_imdb_time(args):
    ##################################################
    # save_prefix = 'data/preprocessed/IMDB_processed_test/'
    raw_save_prefix = 'data/preprocessed/IMDB_processed_time_{}year/'.format(args['time_scale'])
    pathlib.Path(raw_save_prefix).mkdir(parents=True, exist_ok=True)
    num_ntypes = 3


    ##################################################
    # load raw data, delete movies with no actor or director
    print("\n* Loading data.")
    movies = pd.read_csv('data/raw/IMDB/IMDB.csv', encoding='utf-8').dropna(axis=0, subset=['actor_1_name', 'director_name', 'title_year']).sort_values(by = 'title_year', ascending=False).reset_index(drop=True)
    min_year = movies['title_year'].min()
    max_year = movies['title_year'].max()
    time_window = args['time_scale']
    n_snaps = int((max_year - min_year) / time_window)

    
    print("* Extract labels.")
    ##################################################
    # extract labels, and delete movies with unwanted genres
    # 0 for action, 1 for comedy, 2 for drama, -1 for others
    labels_0 = np.zeros((len(movies)), dtype=int)
    for movie_idx, genres in movies['genres'].iteritems():
        labels_0[movie_idx] = -1
        for genre in genres.split('|'):
            if genre == 'Action':
                labels_0[movie_idx] = 0
                break
            elif genre == 'Comedy':
                labels_0[movie_idx] = 1
                break
            elif genre == 'Drama':
                labels_0[movie_idx] = 2
                break
    unwanted_idx = np.where(labels_0 == -1)[0]
    movies = movies.drop(unwanted_idx).reset_index(drop=True)
    labels_0 = np.delete(labels_0, unwanted_idx, 0)


    ##################################################
    # get director list and actor list
    directors = list(set(movies['director_name'].dropna()))
    directors.sort()
    
    actors = list(set(movies['actor_1_name'].dropna().to_list() +
                    movies['actor_2_name'].dropna().to_list() +
                    movies['actor_3_name'].dropna().to_list()))
    actors.sort()
    
    
    ##################################################
    # extract labels, and delete movies with unwanted genres
    # 0 for action, 1 for comedy, 2 for drama, -1 for others
    labels_1 = list(np.zeros((len(directors)), dtype=int))
    for movie_idx, genres in movies['genres'].iteritems():
        director_idx = directors.index(movies.loc[movie_idx]['director_name'])
        labels_1[director_idx] = []
        for genre in genres.split('|'):
            if genre == 'Action':
                labels_1[director_idx].append(0)
            elif genre == 'Comedy':
                labels_1[director_idx].append(1)
            elif genre == 'Drama':
                labels_1[director_idx].append(2)
    

    ##################################################
    # extract labels, and delete movies with unwanted genres
    # 0 for action, 1 for comedy, 2 for drama, -1 for others
    labels_2 = list(np.zeros((len(actors)), dtype=int))
    for movie_idx, genres in movies['genres'].iteritems():
        actor_1_name = movies.loc[movie_idx]['actor_1_name']
        actor_idx1 = -1
        if actor_1_name in actors:
            actor_idx1 = actors.index(movies.loc[movie_idx]['actor_1_name'])
        actor_2_name = movies.loc[movie_idx]['actor_2_name']
        actor_idx2 = -1
        if actor_2_name in actors:
            actor_idx2 = actors.index(movies.loc[movie_idx]['actor_2_name'])
        actor_3_name = movies.loc[movie_idx]['actor_3_name']
        actor_idx3 = -1
        if actor_3_name in actors:
            actor_idx3 = actors.index(movies.loc[movie_idx]['actor_3_name'])
        if actor_idx1 != -1:
            labels_2[actor_idx1] = []
        if actor_idx2 != -1:
            labels_2[actor_idx2] = []
        if actor_idx3 != -1:
            labels_2[actor_idx3] = []
        for genre in genres.split('|'):
            if genre == 'Action':
                if actor_idx1 != -1:
                    labels_2[actor_idx1].append(0)
                if actor_idx2 != -1:
                    labels_2[actor_idx2].append(0)
                if actor_idx3 != -1:
                    labels_2[actor_idx3].append(0)
            elif genre == 'Comedy':
                if actor_idx1 != -1:
                    labels_2[actor_idx1].append(1)
                if actor_idx2 != -1:
                    labels_2[actor_idx2].append(1)
                if actor_idx3 != -1:
                    labels_2[actor_idx3].append(1)
            elif genre == 'Drama':
                if actor_idx1 != -1:
                    labels_2[actor_idx1].append(2)
                if actor_idx2 != -1:
                    labels_2[actor_idx2].append(2)
                if actor_idx3 != -1:
                    labels_2[actor_idx3].append(2)
    
    
    labels = [labels_0, labels_1, labels_2]
    
    
    print("* Building type masks.")
    ##################################################
    # build the adjacency matrix for the graph consisting of movies, directors and actors
    # 0 for movies, 1 for directors, 2 for actors
    dim = len(movies) + len(directors) + len(actors)
    type_mask = np.zeros((dim), dtype=int)
    type_mask[len(movies):len(movies)+len(directors)] = 1
    type_mask[len(movies)+len(directors):] = 2

    
    # -------------------------------------------------------- #
    expected_metapaths = [
        # [(0, 1, 0)],
        # [(0, 1, 0), (0, 2, 0)],
        [(1, 0, 1)],
        [(1, 0, 1), (1, 0, 2, 0, 1)],
        [(2, 0, 2)],
        [(2, 0, 2), (2, 0, 1, 0, 2)]
    ]
    
    task_cnt = 0
    expected_metapaths_cnt = len(expected_metapaths)
    for exp_metapath in expected_metapaths:
        task_cnt += 1
        print("\n* Running Tasks {}/{}.".format(task_cnt, expected_metapaths_cnt))
        target_type = exp_metapath[0][0]
        metapath_num = len(exp_metapath)
        nodes_num = len(labels[target_type])
        metapaths_str = ''.join(map(str, exp_metapath)).replace(" ", "").replace(",", "").replace("(", "_").replace(")", "")
        save_prefix = raw_save_prefix + "metapaths{}/".format(metapaths_str)
        pathlib.Path(save_prefix).mkdir(parents=True, exist_ok=True)
        
        readme_file = open(save_prefix + 'README.txt', mode='w')
        readme_file.writelines("note: 0 for movies, 1 for directors, 2 for actors\n\n")
        readme_file.writelines("target type: {}\n".format(target_type))
        readme_file.writelines("time scale: {}\n".format(time_window))
        readme_file.writelines("snaps: {}\n".format(n_snaps))
        readme_file.writelines("nodes num: {}\n".format(nodes_num))
        readme_file.writelines("metapath num: {}\n".format(metapath_num))
        readme_file.writelines("metapaths: {}\n".format(exp_metapath))
        readme_file.close()
        
        print("~ Splits train/validation/test.")
        # train/validation/test splits
        rand_seed = 1566911444
        train_idx, val_idx = train_test_split(np.arange(len(labels[target_type])), test_size=400, random_state=rand_seed)
        train_idx, test_idx = train_test_split(train_idx, test_size=1000, random_state=rand_seed)
        train_idx.sort()
        val_idx.sort()
        test_idx.sort()
        
        ALL_DATA = {"labels": labels[target_type], "snaps": [], "nodes_num": nodes_num, 
                    "target_type": target_type, "type":["movies", "directors", "actors"], 
                    "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
        
        for snap in range(n_snaps):
            print("~ Processing Tasks {}/{} Snap {}/{}".format(task_cnt, expected_metapaths_cnt, snap + 1, n_snaps))
            
            snap_data = {"adjMs": [], "features": None}
            sta_year = max_year - time_window * snap
            end_year = max_year - time_window * (snap + 1)
            adjMs, features = getSnaps(dim, sta_year, end_year, exp_metapath, movies, directors, actors, type_mask)
            
            snap_data['adjMs'] = adjMs
            snap_data['features'] = features
            ALL_DATA["snaps"].append(snap_data)
        with open(save_prefix + 'all_data_raw.pkl', 'wb') as f:
            pkl.dump(ALL_DATA, f)
        
        # !!!
        print("~ (Tasks {}/{}) Post processing.".format(task_cnt, expected_metapaths_cnt))
        ALL_DATA['snaps'].reverse()

        for snap in ALL_DATA['snaps']:
            valid_nodes = set()
            for adjM in snap['adjMs'].values():
                adj = np.array(adjM.todense())
                line_sum = adj.sum(1)
                valid_nodes_id = np.where(line_sum != 0)[0]
                valid_nodes.update(valid_nodes_id)
            valid_nodes = list(valid_nodes)
            valid_nodes.sort()
            snap['valid_nodes'] = valid_nodes
        
        ALL_DATA['snaps_raw'] = []
        for snap in ALL_DATA['snaps']:
            valid_nodes = snap['valid_nodes']
            feat = np.array(snap['features'].todense())
            feat = feat[valid_nodes,:]
            snap['features'] = sparse.csr_matrix(feat)
            adjMs_raw = {}
            for metapath in snap['adjMs'].keys():
                adjMs_raw[metapath] = snap['adjMs'][metapath]
                adj = np.array(snap['adjMs'][metapath].todense())
                adj = adj[valid_nodes,:][:,valid_nodes]
                snap['adjMs'][metapath] = sparse.csr_matrix(adj)
            ALL_DATA['snaps_raw'].append(adjMs_raw)
        
        with open(save_prefix + 'all_data.pkl', 'wb') as f:
            pkl.dump(ALL_DATA, f)
        print("~ (Tasks {}/{}) All Complete!".format(task_cnt, expected_metapaths_cnt))



def getSnaps(dim, sta_year, end_year, expected_metapath, movies, directors, actors, type_mask):
    adjM = np.zeros((dim, dim), dtype=int)
    for movie_idx, row in movies.iterrows():
        if row['title_year'] <= sta_year and row['title_year'] > end_year:
            if row['director_name'] in directors:
                director_idx = directors.index(row['director_name'])
                adjM[movie_idx, len(movies) + director_idx] = 1
                adjM[len(movies) + director_idx, movie_idx] = 1
            if row['actor_1_name'] in actors:
                actor_idx = actors.index(row['actor_1_name'])
                adjM[movie_idx, len(movies) + len(directors) + actor_idx] = 1
                adjM[len(movies) + len(directors) + actor_idx, movie_idx] = 1
            if row['actor_2_name'] in actors:
                actor_idx = actors.index(row['actor_2_name'])
                adjM[movie_idx, len(movies) + len(directors) + actor_idx] = 1
                adjM[len(movies) + len(directors) + actor_idx, movie_idx] = 1
            if row['actor_3_name'] in actors:
                actor_idx = actors.index(row['actor_3_name'])
                adjM[movie_idx, len(movies) + len(directors) + actor_idx] = 1
                adjM[len(movies) + len(directors) + actor_idx, movie_idx] = 1
    

    ##################################################
    # extract bag-of-word representations of plot keywords for each movie
    # X is a sparse matrix
    vectorizer = CountVectorizer(min_df=2)
    movie_X = vectorizer.fit_transform(movies['plot_keywords'].fillna('').values)
    # assign features to directors and actors as the means of their associated movies' features
    adjM_da2m = adjM[len(movies):, :len(movies)]
    adjM_da2m_normalized = np.diag(1 / adjM_da2m.sum(axis=1)).dot(adjM_da2m)
    director_actor_X = scipy.sparse.csr_matrix(adjM_da2m_normalized).dot(movie_X)
    full_X = scipy.sparse.vstack([movie_X, director_actor_X])


    ##################################################
    target_type = expected_metapath[0][0]
    # create the directories if they do not exist

    # get metapath based neighbor pairs
    neighbor_pairs = utils.preprocess.get_metapath_neighbor_pairs(adjM, type_mask, expected_metapath)

    # construct and save metapath-based adjM
    metapath_adjM_list = utils.preprocess.get_metapath_based_adjM(neighbor_pairs, type_mask, target_type)

    # networkx graph (metapath specific)
    metapath_adjMs = {}
    for G, metapath in zip(metapath_adjM_list, expected_metapath):
        metapath_adjMs[''.join(map(str, metapath))] = scipy.sparse.csr_matrix(G)

    return metapath_adjMs, full_X[np.where(type_mask == target_type)[0]]
