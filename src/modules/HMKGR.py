import torch
import torch.nn as nn
from modules.GraphConv import GraphConv
import torch.nn.functional as F
from utils.util import L2_loss_mean


class HMKGR(nn.Module):
    def __init__(self, args, data):
        super(HMKGR, self).__init__()

        self.n_users = data.n_users
        self.n_items = data.n_items
        self.n_relations = data.n_relations
        self.n_ukg_relations = data.n_ukg_relations

        self.n_entities = data.n_entities
        self.n_nodes = data.n_nodes
        self.device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.dropout = args.node_dropout
        self.dropout_rate = args.node_dropout_rate

        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        self.context_hops = args.context_hops

        self.edge_index, self.edge_type = self.get_edges(data.ckg_graph)
        self.ukg_edge_index, self.ukg_edge_type = self.get_edges(data.ukg_graph)

        self.triplet_item_att = self.triplet_sampling(self.edge_index, self.edge_type).t()

        self.image_embedding = nn.Embedding.from_pretrained(data.image_features, freeze=True)
        self.text_embedding = nn.Embedding.from_pretrained(data.text_features, freeze=True)

        self.item_embedding = nn.Embedding(self.n_items, self.embed_dim)
        self.other_embedding = nn.Embedding(self.n_nodes - self.n_items, self.embed_dim)
        self.relation_embedding = nn.Embedding(self.n_relations, self.relation_dim)

        self.other_embedding_image = nn.Embedding(self.n_nodes - self.n_items, self.embed_dim)
        self.other_embedding_text = nn.Embedding(self.n_nodes - self.n_items, self.embed_dim)
        self.relation_embedding_image = nn.Embedding(self.n_relations, self.relation_dim)
        self.relation_embedding_text = nn.Embedding(self.n_relations, self.relation_dim)

        self.image_linear = nn.Linear(data.image_features.shape[1], self.embed_dim * 4)
        self.image_linear_2 = nn.Linear(4 * self.embed_dim, int(self.embed_dim))
        self.text_linear = nn.Linear(data.text_features.shape[1], self.embed_dim * 4)
        self.text_linear_2 = nn.Linear(4 * self.embed_dim, int(self.embed_dim))

        self.gate1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.gate2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.gate3 = nn.Linear(self.embed_dim, self.embed_dim)
        self.gate4 = nn.Linear(self.embed_dim, self.embed_dim)

        self.sigmoid = nn.Sigmoid()

        self.ukg_relation_embedding_image = nn.Embedding(self.n_ukg_relations, self.relation_dim)
        self.ukg_relation_embedding_text = nn.Embedding(self.n_ukg_relations, self.relation_dim)
        self.ukg_relation_embedding = nn.Embedding(self.n_ukg_relations, self.relation_dim)

        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.other_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

        nn.init.xavier_uniform_(self.ukg_relation_embedding_image.weight)
        nn.init.xavier_uniform_(self.ukg_relation_embedding_text.weight)
        nn.init.xavier_uniform_(self.ukg_relation_embedding.weight)

        nn.init.xavier_uniform_(self.other_embedding_image.weight)
        nn.init.xavier_uniform_(self.other_embedding_text.weight)

        nn.init.xavier_uniform_(self.relation_embedding_image.weight)
        nn.init.xavier_uniform_(self.relation_embedding_text.weight)

        self.gcn = GraphConv(embed_dim=self.embed_dim,
                             n_hops=self.context_hops,
                             n_users=self.n_users,
                             n_relations=self.n_relations,
                             n_items=self.n_items,
                             device=self.device,
                             dropout_rate=self.dropout_rate)

        self.criterion = torch.nn.BCELoss()

    def forward(self, *input, mode):
        if mode == 'ctr':
            return self.calc_ctr_score(*input)
        if mode == 'topK':
            return self.calc_topK_score(*input)
        if mode == 'train':
            return self.calc_score(*input)

    def get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))
        index = graph_tensor[:, :-1]
        type = graph_tensor[:, -1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def triplet_sampling(self, edge_index, edge_type):
        edge_index_t = edge_index.t()
        sample = []
        for idx, h_t in enumerate(edge_index_t):
            if (h_t[0] >= self.n_items and h_t[1] < self.n_items) or (h_t[0] < self.n_items and h_t[1] >= self.n_items):
                sample.append(idx)
        sample = torch.LongTensor(sample)
        return edge_index_t[sample]

    def calc_cf_embeddings(self, user_ids, item_ids):
        image_features = self.image_linear_2(F.leaky_relu(self.image_linear(self.image_embedding.weight)))
        text_features = self.text_linear_2(F.leaky_relu(self.text_linear(self.text_embedding.weight)))

        ego_embed_image = torch.cat((image_features, self.other_embedding_image.weight), dim=0).to(self.device)
        ego_embed_text = torch.cat((text_features, self.other_embedding_text.weight), dim=0).to(self.device)

        all_embed_image = self.gcn(ego_embed_image, self.edge_index, self.edge_type,
                                   self.relation_embedding_image.weight,
                                   dropout=self.dropout)
        all_embed_text = self.gcn(ego_embed_text, self.edge_index, self.edge_type,
                                  self.relation_embedding_text.weight,
                                  dropout=self.dropout)

        user_ego_embed_image = all_embed_image[self.n_entities:]
        final_user_embed_image = self.gcn(user_ego_embed_image, self.ukg_edge_index, self.ukg_edge_type,
                                          self.ukg_relation_embedding_image.weight,
                                          dropout=self.dropout)

        user_ego_embed_text = all_embed_text[self.n_entities:]
        final_user_embed_text = self.gcn(user_ego_embed_text, self.ukg_edge_index, self.ukg_edge_type,
                                         self.ukg_relation_embedding_text.weight,
                                         dropout=self.dropout)

        gi1 = self.sigmoid(
            self.gate1(final_user_embed_image[(user_ids - self.n_entities)]) + self.gate2(all_embed_image[user_ids]))
        gi2 = self.sigmoid(
            self.gate3(final_user_embed_text[(user_ids - self.n_entities)]) + self.gate4(all_embed_text[user_ids]))

        user_f_image = (gi1 * final_user_embed_image[(user_ids - self.n_entities)]) + (
                (1 - gi1) * all_embed_image[user_ids])
        user_f_text = (gi2 * final_user_embed_text[(user_ids - self.n_entities)]) + (
                (1 - gi2) * all_embed_text[user_ids])

        user_embed = torch.cat((user_f_image, user_f_text), dim=1).to(self.device)
        item_embed = torch.cat((all_embed_image[item_ids], all_embed_text[item_ids]), dim=1).to(self.device)

        return user_embed, item_embed

    def calc_score(self, user_ids, item_ids, labels):
        user_embed, item_embed = self.calc_cf_embeddings(user_ids, item_ids)

        logits = torch.sigmoid((user_embed * item_embed).sum(dim=-1)).squeeze()
        cf_loss = self.criterion(logits, labels)

        l2_loss = L2_loss_mean(user_embed) + L2_loss_mean(item_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss

        return loss

    def calc_topK_score(self, user_ids, item_ids):
        user_embed, item_embed = self.calc_cf_embeddings(user_ids, item_ids)

        cf_score = torch.sigmoid((user_embed * item_embed).sum(dim=1)).squeeze()

        return cf_score

    def calc_ctr_score(self, user_ids, item_ids):
        user_embed, item_embed = self.calc_cf_embeddings(user_ids, item_ids)

        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))
        cf_score = torch.sigmoid(cf_score)

        return cf_score
