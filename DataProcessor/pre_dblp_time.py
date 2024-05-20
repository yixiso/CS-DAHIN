import pathlib

import numpy as np
import scipy.sparse
import scipy.io
from scipy import sparse
import pandas as pd
import pickle
import os
import sys
import networkx as nx
import utils.preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer

from utils.data import load_glove_vectors


def pre_dblp_time(args):
    ##################################################
    # save_prefix = 'data/preprocessed/DBLP_processed_time/'
    raw_save_prefix = 'data/preprocessed/DBLP_processed_time_{}year/'.format(args['time_scale'])
    pathlib.Path(raw_save_prefix).mkdir(parents=True, exist_ok=True)
    num_ntypes = 4


    # -------------------------------------------------------- #
    # 作者(author)、论文(paper)、术语(term)以及会议(conference)
    # 0 for authors, 1 for papers, 2 for terms, 3 for conferences
    # expected_metapaths = [
    #     [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)],
    #     [(1, 0, 1), (1, 2, 1), (1, 3, 1)],
    #     [(2, 1, 2), (2, 1, 0, 1, 2), (2, 1, 3, 1, 2)],
    #     [(3, 1, 3), (3, 1, 0, 1, 3), (3, 1, 2, 1, 3)]
    # ]
    expected_metapaths = [
        [(0, 1, 0)],
        [(0, 1, 0), (0, 1, 2, 1, 0)],
        [(0, 1, 2, 1, 0)],
        [(0, 1, 3, 1, 0)],
        [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)]
    ]
    
    # ！！！ Only Label 0 & 1, target_type 0 & 1
    task_cnt = 0
    expected_metapaths_cnt = len(expected_metapaths)
    for exp_metapath in expected_metapaths:
        task_cnt += 1
        print("\n* Running Tasks {}/{}.".format(task_cnt, expected_metapaths_cnt))
        target_type = exp_metapath[0][0]
        metapath_num = len(exp_metapath)
        
        
        ##################################################
        print("~ Loading data.")
        author_label = pd.read_csv('data/raw/DBLP/author_label.txt', sep='\t', header=None, names=['author_id', 'label', 'author_name'], keep_default_na=False, encoding='utf-8')
        # conf_label = pd.read_csv('data/raw/DBLP/conf_label.txt', sep='\t', header=None, names=['conf_id', 'label', 'conf_name'], keep_default_na=False, encoding='utf-8')
        paper_label = pd.read_csv('data/raw/DBLP/paper_label.txt', sep='\t', header=None, names=['paper_id', 'label', 'paper_name'], keep_default_na=False, encoding='utf-8')

        paper_author = pd.read_csv('data/raw/DBLP/paper_author.txt', sep='\t', header=None, names=['paper_id', 'author_id'], keep_default_na=False, encoding='utf-8')
        paper_conf = pd.read_csv('data/raw/DBLP/paper_conf.txt', sep='\t', header=None, names=['paper_id', 'conf_id'], keep_default_na=False, encoding='utf-8')
        paper_term = pd.read_csv('data/raw/DBLP/paper_term.txt', sep='\t', header=None, names=['paper_id', 'term_id'], keep_default_na=False, encoding='utf-8')

        authors = pd.read_csv('data/raw/DBLP/author.txt', sep='\t', header=None, names=['author_id', 'author'], keep_default_na=False, encoding='utf-8')
        papers = pd.read_csv('data/raw/DBLP/paper.txt', sep='\t', header=None, names=['paper_id', 'paper_title'], keep_default_na=False, encoding='cp1252')
        terms = pd.read_csv('data/raw/DBLP/term.txt', sep='\t', header=None, names=['term_id', 'term'], keep_default_na=False, encoding='utf-8')
        confs = pd.read_csv('data/raw/DBLP/conf.txt', sep='\t', header=None, names=['conf_id', 'conf'], keep_default_na=False, encoding='utf-8')
        
        paper_years = pd.read_csv('data/raw/DBLP/year.txt', sep='\t', header=None, names=['paper_id', 'year'], keep_default_na=False, encoding='utf-8')
        
        min_year = paper_years['year'].min()
        max_year = paper_years['year'].max()
        time_window = args['time_scale']
        n_snaps = int((max_year - min_year) / time_window)

        
        ##################################################
        print("~ Filtering data.")
        if target_type == 0:
            # filter out all nodes which does not associated with labeled authors
            labeled_authors = author_label['author_id'].to_list()
            authors = authors[authors['author_id'].isin(labeled_authors)].reset_index(drop=True)
            paper_author = paper_author[paper_author['author_id'].isin(labeled_authors)].reset_index(drop=True)
            
            valid_papers = paper_author['paper_id'].unique()
            papers = papers[papers['paper_id'].isin(valid_papers)].reset_index(drop=True)
            paper_conf = paper_conf[paper_conf['paper_id'].isin(valid_papers)].reset_index(drop=True)
            paper_term = paper_term[paper_term['paper_id'].isin(valid_papers)].reset_index(drop=True)
            
            valid_terms = paper_term['term_id'].unique()
            terms = terms[terms['term_id'].isin(valid_terms)].reset_index(drop=True)
        elif target_type == 1:
            # filter out all nodes which does not associated with labeled authors
            valid_papers = paper_label['author_id'].to_list()
            paper_author = paper_author[paper_author['author_id'].isin(valid_papers)].reset_index(drop=True)
            
            authors = paper_author['author_id'].unique()
            papers = papers[papers['paper_id'].isin(valid_papers)].reset_index(drop=True)
            paper_conf = paper_conf[paper_conf['paper_id'].isin(valid_papers)].reset_index(drop=True)
            paper_term = paper_term[paper_term['paper_id'].isin(valid_papers)].reset_index(drop=True)
            
            valid_terms = paper_term['term_id'].unique()
            terms = terms[terms['term_id'].isin(valid_terms)].reset_index(drop=True)


        ##################################################
        # term lemmatization and grouping
        lemmatizer = WordNetLemmatizer()
        lemma_id_mapping = {}
        lemma_list = []
        lemma_id_list = []
        i = 0
        for _, row in terms.iterrows():
            i += 1
            lemma = lemmatizer.lemmatize(row['term'])
            lemma_list.append(lemma)
            if lemma not in lemma_id_mapping:
                lemma_id_mapping[lemma] = row['term_id']
            lemma_id_list.append(lemma_id_mapping[lemma])
        terms['lemma'] = lemma_list
        terms['lemma_id'] = lemma_id_list

        term_lemma_mapping = {row['term_id']: row['lemma_id'] for _, row in terms.iterrows()}
        lemma_id_list = []
        for _, row in paper_term.iterrows():
            lemma_id_list.append(term_lemma_mapping[row['term_id']])
        paper_term['lemma_id'] = lemma_id_list

        paper_term = paper_term[['paper_id', 'lemma_id']]
        paper_term.columns = ['paper_id', 'term_id']
        paper_term = paper_term.drop_duplicates()
        terms = terms[['lemma_id', 'lemma']]
        terms.columns = ['term_id', 'term']
        terms = terms.drop_duplicates()


        ##################################################
        # filter out stopwords from terms
        stopwords = sklearn_stopwords.union(set(nltk_stopwords.words('english')))
        stopword_id_list = terms[terms['term'].isin(stopwords)]['term_id'].to_list()
        paper_term = paper_term[~(paper_term['term_id'].isin(stopword_id_list))].reset_index(drop=True)
        terms = terms[~(terms['term'].isin(stopwords))].reset_index(drop=True)



        ##################################################
        author_label = author_label.sort_values('author_id').reset_index(drop=True)
        authors = authors.sort_values('author_id').reset_index(drop=True)
        papers = papers.sort_values('paper_id').reset_index(drop=True)
        terms = terms.sort_values('term_id').reset_index(drop=True)
        confs = confs.sort_values('conf_id').reset_index(drop=True)


        ##################################################
        print("~ Extracting labels.")
        # extract labels
        if target_type == 0:
            labels = author_label['label'].to_numpy()
        elif target_type == 1:
            labels = paper_label['label'].to_numpy()

        
        ##################################################
        print("~ Extracting features.")
        # extract features
        if target_type == 0:
            # use HAN paper's preprocessed data as the features of authors (https://github.com/Jhy1993/HAN)
            mat = scipy.io.loadmat('data/raw/DBLP/DBLP4057_GAT_with_idx.mat')
            features_author = np.array(list(zip(*sorted(zip(labeled_authors, mat['features']), key=lambda tup: tup[0])))[1])
            features = scipy.sparse.csr_matrix(features_author)
        elif target_type == 1:
            # use bag-of-words representation of paper titles as the features of papers
            class LemmaTokenizer:
                def __init__(self):
                    self.wnl = WordNetLemmatizer()
                def __call__(self, doc):
                    return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
            vectorizer = CountVectorizer(min_df=2, stop_words=stopwords, tokenizer=LemmaTokenizer())
            features = vectorizer.fit_transform(papers['paper_title'].values)
            
            
        print("~ Building type masks.")
        ##################################################
        # 作者(author)、论文(paper)、术语(term)以及会议(conference)
        # build the adjacency matrix for the graph consisting of authors, papers, terms and conferences
        # 0 for authors, 1 for papers, 2 for terms, 3 for conferences
        dim = len(authors) + len(papers) + len(terms) + len(confs)
        type_mask = np.zeros((dim), dtype=int)
        type_mask[len(authors):len(authors)+len(papers)] = 1
        type_mask[len(authors)+len(papers):len(authors)+len(papers)+len(terms)] = 2
        type_mask[len(authors)+len(papers)+len(terms):] = 3

        author_id_mapping = {row['author_id']: i for i, row in authors.iterrows()}
        paper_id_mapping = {row['paper_id']: i + len(authors) for i, row in papers.iterrows()}
        term_id_mapping = {row['term_id']: i + len(authors) + len(papers) for i, row in terms.iterrows()}
        conf_id_mapping = {row['conf_id']: i + len(authors) + len(papers) + len(terms) for i, row in confs.iterrows()}

        
        # -------------------------------------------------------- #
        nodes_num = len(labels)
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
        # author train/validation/test splits
        rand_seed = 1566911444
        train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=400, random_state=rand_seed)
        train_idx, test_idx = train_test_split(train_idx, test_size=3257, random_state=rand_seed)
        train_idx.sort()
        val_idx.sort()
        test_idx.sort()
        
        ALL_DATA = {"labels": labels, "snaps": [], "nodes_num": nodes_num, 
                    "target_type": target_type, "type":["authors", "papers", "terms", "conferences"], 
                    "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
            
        for snap in range(n_snaps):
            print("~ Processing Tasks {}/{} Snap {}/{}".format(task_cnt, expected_metapaths_cnt, snap + 1, n_snaps))
            
            snap_data = {"adjMs": [], "features": None}
            sta_year = max_year - time_window * snap
            end_year = max_year - time_window * (snap + 1)
            adjMs = getSnaps(dim, sta_year, end_year, exp_metapath, type_mask, paper_years, paper_author, paper_term, paper_conf, paper_id_mapping, author_id_mapping, term_id_mapping, conf_id_mapping)
            
            snap_data['adjMs'] = adjMs
            snap_data['features'] = features
            ALL_DATA["snaps"].append(snap_data)
        with open(save_prefix + 'all_data_raw.pkl', 'wb') as f:
            pickle.dump(ALL_DATA, f)
        
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
            pickle.dump(ALL_DATA, f)
        print("~ (Tasks {}/{}) All Complete!".format(task_cnt, expected_metapaths_cnt))


def getSnaps(dim, sta_year, end_year, expected_metapath, type_mask, paper_years, paper_author, paper_term, paper_conf, paper_id_mapping, author_id_mapping, term_id_mapping, conf_id_mapping):
    adjM = np.zeros((dim, dim), dtype=int)
    print("- Preparing adjM.")
    print("- (1/3) Parsing subgraph paper_author.")
    for i, row in paper_author.iterrows():
        p_year = int(paper_years[paper_years['paper_id'] == row['paper_id']]['year'])
        if p_year <= sta_year and p_year > end_year:
            idx1 = paper_id_mapping[row['paper_id']]
            idx2 = author_id_mapping[row['author_id']]
            adjM[idx1, idx2] = 1
            adjM[idx2, idx1] = 1
    print("- (2/3) Parsing subgraph paper_term.")
    for i, row in paper_term.iterrows():
        p_year = int(paper_years[paper_years['paper_id'] == row['paper_id']]['year'])
        if p_year <= sta_year and p_year > end_year:
            idx1 = paper_id_mapping[row['paper_id']]
            idx2 = term_id_mapping[row['term_id']]
            adjM[idx1, idx2] = 1
            adjM[idx2, idx1] = 1
    print("- (3/3) Parsing subgraph paper_conf.")
    for i, row in paper_conf.iterrows():
        p_year = int(paper_years[paper_years['paper_id'] == row['paper_id']]['year'])
        if p_year <= sta_year and p_year > end_year:
            idx1 = paper_id_mapping[row['paper_id']]
            idx2 = conf_id_mapping[row['conf_id']]
            adjM[idx1, idx2] = 1
            adjM[idx2, idx1] = 1

    ##################################################
    target_type = expected_metapath[0][0]
    # create the directories if they do not exist
    neighbor_pairs = utils.preprocess.get_metapath_neighbor_pairs(adjM, type_mask, expected_metapath)
    # construct and save metapath-based adjM
    metapath_adjM_list = utils.preprocess.get_metapath_based_adjM(neighbor_pairs, type_mask, target_type)
    
    # networkx graph (metapath specific)
    metapath_adjMs = {}
    for G, metapath in zip(metapath_adjM_list, expected_metapath):
        metapath_adjMs[''.join(map(str, metapath))] = scipy.sparse.csr_matrix(G)

    return metapath_adjMs
