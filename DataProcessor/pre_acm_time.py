import pathlib
import scipy.io
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle as pkl
from scipy import sparse
import copy


data_prefix = "./data/raw/ACM/"
save_prefix = "./data/preprocessed/ACM_processed_time/"


def pre_acm_time(args):
    global save_prefix
    save_prefix = 'data/preprocessed/ACM_processed_time_{}year/metapaths_AFA/'.format(args['time_scale'])
    pathlib.Path(save_prefix).mkdir(parents=True, exist_ok=True)
    # if args['use_cache'] == 'none':
    #     args['use_cache'] = choose_use_cache()
    # elif args['use_cache'] in ["Yes", "yes", "y", "Y"]:
    #     args['use_cache'] = True
    # elif args['use_cache'] in ["No", "no", "N", "n"]:
    #     args['use_cache'] = False
    
    # if not args['use_cache']:
    #     preprocess_acm()
    
    preprocess_acm(args)
    divide_time_snaps(args)


def choose_use_cache():
    while True:
        ch = input("Do you want to use cached data? [Yes(default)/No]:")
        if ch in ["Yes", "yes", "y", "Y", ""]:
            return True
        elif ch in ["No", "no", "N", "n"]:
            return False
        else:
            print("Options are illegal, please input again.")


def preprocess_acm(args):
    print("* Preprocess ACM.")
    ACM_RAW = scipy.io.loadmat(data_prefix + 'ACM.mat')
    years_C = np.loadtxt(data_prefix + 'year.txt')
    papers = ACM_RAW['P']
    PvsC = ACM_RAW['PvsC']

    valid_papers_ind = []
    valid_papers = []
    keywords = []
    l = len(papers)
    for i in range(len(papers)):
        if PvsC[i, 0]==1.0 or PvsC[i, 1]==1.0 or PvsC[i,9]==1.0 or PvsC[i, 10]==1.0 or PvsC[i, 13]==1.0:
            valid_papers_ind.append(i)
            valid_papers.append(papers[i][0][0])

    PvsA = ACM_RAW['PvsA'][valid_papers_ind].todense()
    author_num_per_paper = PvsA.sum(1)
    valid_papers_ind = np.array(valid_papers_ind)
    valid_papers_ind = list(valid_papers_ind[list(np.where(author_num_per_paper != 0)[0])])
    valid_papers = np.array(ACM_RAW['P'])[valid_papers_ind]
    valid_papers = [a[0][0] for a in valid_papers]
    valid_authors_ind = np.where(PvsA.sum(0) != 0)[1]
    valid_authors = ACM_RAW['A'][valid_authors_ind]
    valid_authors = [a[0][0] for a in valid_authors]

    AvsF = ACM_RAW['AvsF'][valid_authors_ind].todense()
    valid_foundations_ind = np.where(AvsF.sum(0) != 0)[1]
    valid_foundations = ACM_RAW['F'][valid_foundations_ind]
    valid_foundations = [a[0][0] for a in valid_foundations]
    
    PvsC = ACM_RAW['PvsC'][valid_papers_ind].todense()
    valid_conferences_ind = np.where(PvsC.sum(0) != 0)[1]
    valid_conferences = ACM_RAW['C'][valid_conferences_ind]
    valid_conferences = [a[0][0] for a in valid_conferences]
    
    PvsV = ACM_RAW['PvsV'][valid_papers_ind].todense()
    valid_venue_ind = np.where(PvsV.sum(0) != 0)[1]
    valid_venue = ACM_RAW['V'][valid_venue_ind]
    valid_venue = [a[0][0] for a in valid_venue]
    valid_years = years_C[valid_venue_ind]
    
    PvsA = ACM_RAW['PvsA'][valid_papers_ind, :][:, valid_authors_ind]
    AvsF = ACM_RAW['AvsF'][valid_authors_ind, :][:, valid_foundations_ind]
    PvsC = ACM_RAW['PvsC'][valid_papers_ind, :][:, valid_conferences_ind]
    PvsV = ACM_RAW['PvsV'][valid_papers_ind, :][:, valid_venue_ind]
    

    # extract author labels and features
    # ind: 0 KDD, 1 SIGMOD, 9 SIGCOMM, 10 MobiCOMM, 13 VLDB
    # label: 0 KDD VLDB, 1 SIGCOMM MobiCOMM, 2 SIGMOD
    labels_A = []
    labels_text = [0, 2, 1, 1, 0]
    keywords_A = []
    stopwords = [t[0][0] for t in ACM_RAW['stopwords']]
    theme = [t[0][0] for t in ACM_RAW['T']]
    i = 0
    for author_ind in valid_authors_ind:
        i += 1
        paper_inds = ACM_RAW['PvsA'][:, author_ind].todense()
        paper_inds = paper_inds.A.reshape(-1)
        paper_inds = np.where(paper_inds != 0)[0]
        # conf 0  1  9 10  13
        cnt = np.array([0, 0, 0, 0, 0])
        keywords = ""
        for paper_ind in paper_inds:
            if paper_ind in valid_papers_ind:
                conference_ind = ACM_RAW['PvsC'][paper_ind].todense()
                conference_ind = np.where(conference_ind != 0)[1]
                if conference_ind == 0:
                    cnt[0] += 1
                elif conference_ind == 1:
                    cnt[1] += 1
                elif conference_ind == 9:
                    cnt[2] += 1
                elif conference_ind == 10:
                    cnt[3] += 1
                elif conference_ind == 13:
                    cnt[4] += 1
                # extract keywords
                theme_inds = ACM_RAW['TvsP'][:, paper_ind].todense()
                theme_inds = theme_inds.A.reshape(-1)
                theme_inds = np.where(theme_inds != 0)[0]
                for theme_ind in theme_inds:
                    word = theme[theme_ind].lower()
                    if word not in stopwords:
                        keywords = keywords + "|" + word
        labels_A.append(labels_text[cnt.argmax()])
        keywords = keywords[1:]
        keywords_A.append(keywords)
        print("\r|| {}/{} |{:<50}||".format(i, len(valid_authors_ind), "="*int((i / len(valid_authors_ind))*49+1)), end="")

    vectorizer = CountVectorizer(min_df=2)
    author_X = vectorizer.fit_transform(keywords_A)

    ACM_PRE = {"P": valid_papers, "A": valid_authors, "F": valid_foundations, "C": valid_conferences, "Y": valid_years, 
               "PvsC": PvsC, "PvsA": PvsA, "AvsF": AvsF, "PvsY": PvsV, 
               "features_A": author_X, "labels_A": labels_A}

    with open(save_prefix + 'ACM_PRE.pkl', 'wb') as f:
        pkl.dump(ACM_PRE, f)
    
    print("\nCompelete!\n")


def divide_time_snaps(args):
    # divide time snaps
    print("* Divide time snaps.")
    with open(save_prefix + "ACM_PRE.pkl", "rb") as f:
        ACM_PRE = pkl.load(f)

    labels_A = ACM_PRE['labels_A']
    features_A = ACM_PRE['features_A']
    nodes_num = len(labels_A)

    print("* Splits train/validation/test.")
    # train/validation/test splits
    rand_seed = 1566911444
    train_idx, val_idx = train_test_split(np.arange(len(labels_A)), test_size=400, random_state=rand_seed)
    train_idx, test_idx = train_test_split(train_idx, test_size=1000, random_state=rand_seed)
    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    target_type = 0
    ALL_DATA = {"labels": labels_A, "snaps": [], "nodes_num": nodes_num, 
                "target_type": target_type, "type":["authors", "papers", "foundations"], 
                "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}

    min_year = ACM_PRE['Y'].min()
    max_year = ACM_PRE['Y'].max()
    time_window = args['time_scale']
    n_snaps = int((max_year - min_year) / time_window)
    
    readme_file = open(save_prefix + 'README.txt', mode='w')
    readme_file.writelines("note: 0 for authors, 1 for papers, 2 for foundations\n\n")
    readme_file.writelines("target type: {}\n".format(target_type))
    readme_file.writelines("time scale: {}\n".format(time_window))
    readme_file.writelines("snaps: {}\n".format(n_snaps))
    readme_file.writelines("nodes num: {}\n".format(nodes_num))
    readme_file.writelines("metapath num: {}\n".format(2))
    readme_file.writelines("metapaths: {}\n".format("[(APA), (AFA)]"))
    readme_file.close()

    for snap in range(n_snaps):
        snap_data = {"adjMs": [], "features": None}
        sta_year = max_year - time_window * snap
        end_year = max_year - time_window * (snap + 1)
        
        print("~ Processing Snap {}/{} with time {}~{}".format(snap + 1, n_snaps, sta_year, end_year))
        adjMs = getSnaps(ACM_PRE, sta_year, end_year)
        
        snap_data['adjMs'] = adjMs
        snap_data['features'] = features_A
        ALL_DATA["snaps"].append(snap_data)

    with open(save_prefix + 'all_data_raw.pkl', 'wb') as f:
        pkl.dump(ALL_DATA, f)

    # !!!
    print("~ Post processing.")
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
    print("~ All Complete!")


def getSnaps(ACM_PRE, sta_year, end_year):
    valid_years_ind = []
    for ind in range(len(ACM_PRE['Y'])):
        if ACM_PRE['Y'][ind] <= sta_year and ACM_PRE['Y'][ind] > end_year:
            valid_years_ind.append(ind)

    valid_papers_ind = ACM_PRE['PvsY'][:, valid_years_ind].sum(1)
    valid_papers_ind = np.where(valid_papers_ind != 0)[0]
    invalid_papers_ind = set(range(len(ACM_PRE['P'])))
    invalid_papers_ind = list(invalid_papers_ind - set(valid_papers_ind))
    
    valid_authors_ind = ACM_PRE['PvsA'][valid_papers_ind].sum(0)
    valid_authors_ind = np.where(valid_authors_ind != 0)[1]
    invalid_authors_ind = set(range(len(ACM_PRE['A'])))
    invalid_authors_ind = list(invalid_authors_ind - set(valid_authors_ind))
    
    valid_foundations_ind = ACM_PRE['AvsF'][valid_authors_ind].sum(0)
    valid_foundations_ind = np.where(valid_foundations_ind != 0)[1]
    invalid_foundations_ind = set(range(len(ACM_PRE['F'])))
    invalid_foundations_ind = list(invalid_foundations_ind - set(valid_foundations_ind))

    PA = ACM_PRE['PvsA'].todense()
    PA[invalid_papers_ind] = 0
    PA[:, invalid_authors_ind] = 0
    APA = PA.T * PA

    AF = ACM_PRE['AvsF'].todense()
    AF[invalid_authors_ind] = 0
    AF[:, invalid_foundations_ind] = 0
    AFA = AF * AF.T
    
    for i in range(APA.shape[0]):
        for j in range(APA.shape[1]):
            if APA[i, j] > 0:
                APA[i, j] = 1
            if AFA[i, j] > 0:
                AFA[i, j] = 1
        print("\r|| {}/{} |{:<50}||".format(i, APA.shape[0], "="*int((i / APA.shape[0])*49+1)), end="")
    print("")

    # metapath_adjMs = {"APA": sparse.csr_matrix(APA), "AFA": sparse.csr_matrix(AFA)}
    # metapath_adjMs = {"APA": sparse.csr_matrix(APA)}
    metapath_adjMs = {"AFA": sparse.csr_matrix(AFA)}

    return metapath_adjMs