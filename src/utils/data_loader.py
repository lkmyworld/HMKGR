import logging
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import networkx as nx
import scipy.sparse as sp
import torch
import pickle
from tqdm import tqdm


class Dataloader(object):
    def __init__(self, args, logging):
        self.args = args
        self.dataset = args.dataset
        self.dataset_dir = os.path.join(args.data_dir, args.dataset)

        self.ckg_file = os.path.join(self.dataset_dir, "kg_final.txt")
        self.ukg_file = os.path.join(self.dataset_dir, "ukg_final.txt")
        self.pkl_file = os.path.join(self.dataset_dir, 'image_text_pair.pkl')

        self.train_data_with_neg = self.load_data_with_neg(
            os.path.join(self.dataset_dir, 'train_valid_test/train_data_with_neg.pkl'))
        self.valid_data_with_neg = self.load_data_with_neg(
            os.path.join(self.dataset_dir, 'train_valid_test/valid_data_with_neg.pkl'))
        self.test_data_with_neg = self.load_data_with_neg(
            os.path.join(self.dataset_dir, 'train_valid_test/test_data_with_neg.pkl'))

        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_data_with_neg)
        self.cf_valid_data, self.valid_user_dict = self.load_cf(self.valid_data_with_neg)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_data_with_neg)

        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])

        self.test_batch_size = args.test_batch_size

        ckg_data = self.load_ckg()
        self.ckg_graph, self.relation_dict = self.construct_ckg(ckg_data, logging)
        ukg_data = self.load_ukg()
        self.ukg_graph, self.social_relation_dict = self.construct_ukg(ukg_data, logging)

        self.adj_mat_list = self.build_sparse_relational_graph(self.relation_dict)

        self.load_multi_modal(logging)

        self.train_cf_pairs = torch.LongTensor(
            np.array([[u_id, i_id] for u_id, i_id in zip(self.cf_train_data[0], self.cf_train_data[1])], np.int32))
        self.test_cf_pairs = torch.LongTensor(
            np.array([[u_id, i_id] for u_id, i_id in zip(self.cf_test_data[0], self.cf_test_data[1])], np.int32))

        self.print_info(logging)

    def load_data_with_neg(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def load_cf(self, data):
        user = []
        item = []
        user_dict = defaultdict(list)
        for d in data:
            if d[2] == 1:
                user.append(d[0])
                item.append(d[1])
                user_dict[d[0]].append(d[1])
        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict

    def load_ckg(self):
        kg_data = pd.read_csv(self.ckg_file, sep='\t', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def load_ukg(self):
        kg_data = pd.read_csv(self.ukg_file, sep='\t', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def construct_ckg(self, kg_data, logging):
        # add inverse relation
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # add bi-interactions relation
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_nodes = self.n_users + self.n_entities

        self.cf_train_data = (
            np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32),
            self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32),
                             self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = defaultdict(list, {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                                                  self.train_user_dict.items()})
        self.test_user_dict = defaultdict(list, {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                                                 self.test_user_dict.items()})
        self.valid_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                                self.valid_user_dict.items()}

        self.train_data_with_neg[:, 0] += self.n_entities
        self.valid_data_with_neg[:, 0] += self.n_entities
        self.test_data_with_neg[:, 0] += self.n_entities

        # add bi-interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]
        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)

        self.n_kg_train = len(self.kg_train_data)
        self.n_relations = max(self.kg_train_data['r']) + 1

        ckg_graph = nx.MultiDiGraph()
        logging.info("begin load ckg triples ...")
        rd = defaultdict(list)
        for row in self.kg_train_data.iterrows():
            head, relation, tail = row[1]
            ckg_graph.add_edge(head, tail, key=relation)
            rd[relation].append([head, tail])
        return ckg_graph, rd

    def construct_ukg(self, kg_data, logging):

        self.ukg_train_data = kg_data
        self.n_ukg_train = len(self.ukg_train_data)
        self.n_ukg_relations = max(kg_data['r']) + 1

        ukg_graph = nx.MultiDiGraph()
        logging.info("begin load ukg triples ...")
        rd = defaultdict(list)
        for row in self.ukg_train_data.iterrows():
            head, relation, tail = row[1]
            ukg_graph.add_edge(head, tail, key=relation)
            rd[relation].append([head, tail])
        return ukg_graph, rd

    def build_sparse_relational_graph(self, relation_dict):
        adj_mat_list = []
        for r_id in relation_dict.keys():
            np_mat = np.array(relation_dict[r_id])
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(self.n_nodes, self.n_nodes))
            adj_mat_list.append(adj)
        return adj_mat_list

    def load_multi_modal(self, logging):
        logging.info('begin load image_text_pair ...')
        with open(self.pkl_file, 'rb') as f:
            image_text_pair = pickle.load(f)
        image_features = image_text_pair[0]
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = image_text_pair[1]
        text_features /= text_features.norm(dim=-1, keepdim=True)

        self.image_features = image_features
        self.text_features = text_features

    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_nodes:           %d' % self.n_nodes)
        logging.info('n_relations:       %d' % self.n_relations)
        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)
        logging.info('n_kg_train:        %d' % self.n_kg_train)
