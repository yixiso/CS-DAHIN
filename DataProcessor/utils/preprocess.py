import numpy as np
import scipy.sparse
import networkx as nx


def get_metapath_adjacency_matrix(adjM, type_mask, metapath):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param metapath
    :return: a list of metapath-based adjacency matrices
    """
    out_adjM = scipy.sparse.csr_matrix(adjM[np.ix_(type_mask == metapath[0], type_mask == metapath[1])])
    for i in range(1, len(metapath) - 1):
        out_adjM = out_adjM.dot(scipy.sparse.csr_matrix(adjM[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])]))
    return out_adjM.toarray()


# networkx.has_path may search too
def get_metapath_neighbor_pairs(M, type_mask, expected_metapaths):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param expected_metapaths: a list of expected metapaths
    :return: a list of python dictionaries, consisting of metapath-based neighbor pairs and intermediate paths
    """
    mp_len = len(expected_metapaths)
    outs = []
    mp_cnt = 1
    for metapath in expected_metapaths:
        print("- ({}/{}) Finding metapath {}...".format(mp_cnt, mp_len, ''.join(map(str, metapath))))
        # consider only the edges relevant to the expected metapath
        mask = np.zeros(M.shape, dtype=bool)
        for i in range((len(metapath) - 1) // 2):
            temp = np.zeros(M.shape, dtype=bool)
            temp[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])] = True
            temp[np.ix_(type_mask == metapath[i + 1], type_mask == metapath[i])] = True
            mask = np.logical_or(mask, temp)
        partial_g_nx = nx.from_numpy_matrix((M * mask).astype(int))

        # only need to consider the former half of the metapath
        # e.g., we only need to consider 0-1-2 for the metapath 0-1-2-1-0
        metapath_to_target = {}
        source_ind = (type_mask == metapath[0]).nonzero()[0]
        source_num = len(source_ind)
        source_cnt = 1
        for source in (type_mask == metapath[0]).nonzero()[0]:
            for target in (type_mask == metapath[(len(metapath) - 1) // 2]).nonzero()[0]:
                # check if there is a possible valid path from source to target node
                has_path = False
                single_source_paths = nx.single_source_shortest_path(
                    partial_g_nx, source, cutoff=(len(metapath) + 1) // 2 - 1)
                if target in single_source_paths:
                    has_path = True

                #if nx.has_path(partial_g_nx, source, target):
                if has_path:
                    shortests = [p for p in nx.all_shortest_paths(partial_g_nx, source, target) if
                                 len(p) == (len(metapath) + 1) // 2]
                    if len(shortests) > 0:
                        metapath_to_target[target] = metapath_to_target.get(target, []) + shortests
            source_cnt += 1
            print("\r|| source: {}/{} |{:<50}||".format(source_cnt, source_num, "="*int((source_cnt / source_num)*49+1)), end="")
        print("\n- ({}/{}) Finding metapath neighbor paris ...".format(mp_cnt, mp_len))
        metapath_neighbor_paris = {}
        mnp_len = len(metapath_to_target)
        mnp_cnt = 1
        for key, value in metapath_to_target.items():
            for p1 in value:
                for p2 in value:
                    metapath_neighbor_paris[(p1[0], p2[0])] = metapath_neighbor_paris.get((p1[0], p2[0]), []) + [p1 + p2[-2::-1]]
            mnp_cnt += 1
            print("\r|| Finding metapath neighbor paris || {}/{} |{:<50}||".format(mnp_cnt, mnp_len, "="*int((mnp_cnt / mnp_len)*49+1)), end="")
        outs.append(metapath_neighbor_paris)
        mp_cnt += 1
        print("")
    return outs


def get_metapath_based_adjM(neighbor_pairs, type_mask, ctr_ntype):
    indices = np.where(type_mask == ctr_ntype)[0]
    idx_mapping = {}
    for i, idx in enumerate(indices):
        idx_mapping[idx] = i
    adjM_list = []
    mp_process = 0
    for metapaths in neighbor_pairs:
        mp_process += 1
        print("- {}/{} Get metapath based adjM...".format(mp_process, len(neighbor_pairs)))
        edge_count = 0
        sorted_metapaths = sorted(metapaths.items())
        adjM = np.zeros((len(indices), len(indices)), dtype=int)
        process = 0
        for (src, dst), paths in sorted_metapaths:
            process += 1
            for _ in range(len(paths)):
                # G.add_edge(range(len(indices)), idx_mapping[dst])
                adjM[idx_mapping[src], idx_mapping[dst]] += 1
                edge_count += 1
            print("\r|| src: {} dst: {} || {}/{} |{:<50}||".format(src, dst, process, len(sorted_metapaths), "="*int((process / len(sorted_metapaths))*49+1)), end="")
        adjM_list.append(adjM)
        print("")
    return adjM_list


def get_networkx_graph(neighbor_pairs, type_mask, ctr_ntype):
    indices = np.where(type_mask == ctr_ntype)[0]
    idx_mapping = {}
    for i, idx in enumerate(indices):
        idx_mapping[idx] = i
    G_list = []
    for metapaths in neighbor_pairs:
        edge_count = 0
        sorted_metapaths = sorted(metapaths.items())
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(len(indices)))
        for (src, dst), paths in sorted_metapaths:
            for _ in range(len(paths)):
                # G.add_edge(range(len(indices)), idx_mapping[dst])
                G.add_edge(idx_mapping[src], idx_mapping[dst])
                edge_count += 1
        G_list.append(G)
    return G_list


def get_edge_metapath_idx_array(neighbor_pairs):
    all_edge_metapath_idx_array = []
    for metapath_neighbor_pairs in neighbor_pairs:
        sorted_metapath_neighbor_pairs = sorted(metapath_neighbor_pairs.items())
        edge_metapath_idx_array = []
        for _, paths in sorted_metapath_neighbor_pairs:
            edge_metapath_idx_array.extend(paths)
        edge_metapath_idx_array = np.array(edge_metapath_idx_array, dtype=int)
        all_edge_metapath_idx_array.append(edge_metapath_idx_array)
        print(edge_metapath_idx_array.shape)
    return all_edge_metapath_idx_array
