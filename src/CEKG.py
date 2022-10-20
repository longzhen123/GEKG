import time
import numpy as np
import torch.nn as nn
import torch as t
from torch import optim
from sklearn.metrics import roc_auc_score, accuracy_score
from src.evaluate import get_hit, get_ndcg
from src.load_base import load_data, get_records


class GEKG(nn.Module):

    def __init__(self, args, n_entity, n_relation):
        super(GEKG, self).__init__()
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        self.dim = args.dim
        entity_embedding_matrix = t.randn(n_entity, self.dim)
        relation_embedding_matrix = t.randn(n_relation, self.dim)
        nn.init.xavier_uniform_(entity_embedding_matrix)
        nn.init.xavier_uniform_(relation_embedding_matrix)
        self.entity_embedding_matrix = nn.Parameter(entity_embedding_matrix)
        self.relation_embedding_matrix = nn.Parameter(relation_embedding_matrix)

        self.weight_generator = nn.Linear(2 * self.dim, self.dim)
        self.weight_attention = nn.Linear(2 * self.dim, 1)

        self.criterion = nn.BCELoss()

    def forward(self, pairs, item_neighbors, user_records):
        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]

        user_embeddings = self.get_user_embedding(users, user_records)
        item_embeddings, _ = self.get_item_embedding(items, item_neighbors)

        return t.sigmoid((user_embeddings * item_embeddings).sum(dim=1))

    def get_user_embedding(self, users, user_records):

        user_embedding_list = []

        for user in users:
            user_embedding_list.append(self.entity_embedding_matrix[user_records[user]].sum(dim=0).reshape(1, self.dim))

        return t.cat(user_embedding_list, dim=0)

    def get_item_embedding(self, items, item_neighbors):

        entity_list, relation_list = self.get_neighbor(items, item_neighbors)

        entity_embeddings = self.get_entity_embedding(entity_list)
        relation_embeddings = self.get_relation_embedding(relation_list)

        e_r = t.cat([entity_embeddings, relation_embeddings], dim=-1)

        generator_entity_embeddings = t.sigmoid(self.weight_generator(e_r))

        weights_1 = t.sigmoid(self.weight_attention(e_r))
        normalize_weights_1 = t.softmax(weights_1, dim=1)
        neighbor_embeddings = (entity_embeddings * normalize_weights_1).sum(dim=1)

        weights_2 = t.sigmoid(self.weight_attention(t.cat([generator_entity_embeddings, relation_embeddings], dim=-1)))
        normalize_weights_2 = t.softmax(weights_2, dim=1)
        generator_neighbor_embeddings = (generator_entity_embeddings * normalize_weights_2).sum(dim=1)

        generator_loss = (neighbor_embeddings.reshape(-1, self.dim) - generator_neighbor_embeddings.reshape(-1, self.dim)) ** 2

        # return self.entity_embedding_matrix[items] + neighbor_embeddings, 0
        return self.entity_embedding_matrix[items] + neighbor_embeddings + generator_neighbor_embeddings, generator_loss.sum()

    def get_neighbor(self, items, item_neighbors):

        entity_list = []
        relation_list = []

        for item in items:
            entity_list.append(item_neighbors[item][0])
            relation_list.append(item_neighbors[item][1])

        return entity_list, relation_list

    def get_entity_embedding(self, entity_list):

        embedding_list = []

        for list_1 in entity_list:
            embedding_list.append(self.entity_embedding_matrix[list_1].reshape(1, -1, self.dim))

        return t.cat(embedding_list, dim=0)

    def get_relation_embedding(self, relation_list):

        embedding_list = []

        for list_1 in relation_list:
            embedding_list.append(self.relation_embedding_matrix[list_1].reshape(1, -1, self.dim))

        return t.cat(embedding_list, dim=0)

    def cal_loss(self, pairs, item_neighbors, user_records):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]

        user_embeddings = self.get_user_embedding(users, user_records)
        item_embeddings, generator_loss = self.get_item_embedding(items, item_neighbors)
        label = t.tensor([pair[2] for pair in pairs]).float()

        if t.cuda.is_available():
            label = label.to(user_embeddings.device)

        predict = t.sigmoid((user_embeddings * item_embeddings).sum(dim=1))
        base_loss = self.criterion(predict, label)

        return base_loss + 1e-6 * generator_loss


def eval_topk(model, rec, item_neighbors, user_records, batch_size, topk):
    HR, NDCG = [], []
    model.eval()
    for user in rec:

        items = list(rec[user])
        pairs = [[user, item] for item in items]
        predict = []
        for i in range(0, len(pairs), batch_size):
            predict.extend(model.forward(pairs[i: i+batch_size], item_neighbors, user_records).cpu().reshape(-1).detach().numpy().tolist())
        # print(predict)
        n = len(pairs)
        item_scores = {items[i]: predict[i] for i in range(n)}
        item_list = list(dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)).keys())[: topk]
        HR.append(get_hit(items[-1], item_list))
        NDCG.append(get_ndcg(items[-1], item_list))

    model.train()
    return np.mean(HR), np.mean(NDCG)


def eval_ctr(model, pairs, item_neighbors, user_records, batch_size):

    model.eval()
    pred_label = []
    for i in range(0, len(pairs), batch_size):
        batch_label = model(pairs[i: i+batch_size], item_neighbors, user_records).cpu().detach().numpy().tolist()
        pred_label.extend(batch_label)
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return auc, acc


def get_user_records(train_records, K_u):

    user_records = dict()

    for user in train_records:
        records = train_records[user]

        if len(records) > K_u:
            indices = np.random.choice(len(records), K_u, replace=False)
        else:
            indices = np.random.choice(len(records), K_u, replace=True)

        user_records[user] = [records[i] for i in indices]

    return user_records


def get_item_neighbors(items, kg_dict, K_v):

    item_neighbors = {item: [] for item in items}

    for item in items:

        neighbors = kg_dict[item]

        if len(neighbors) >= K_v:
            indices = np.random.choice(len(neighbors), K_v)
        else:
            indices = np.random.choice(len(neighbors), K_v, replace=True)

        item_neighbors[item].append([neighbors[i][1] for i in indices])
        item_neighbors[item].append([neighbors[i][0] for i in indices])

    return item_neighbors


def train(args, is_topk=False):
    np.random.seed(123)

    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, eval_set, test_set, rec, kg_dict = data[4], data[5], data[6], data[7], data[8]
    train_records = get_records(train_set)
    ripple_sets = get_item_neighbors(range(n_item), kg_dict, args.K_v)

    user_records = get_user_records(train_records, args.K_u)
    model = GEKG(args, n_entity, n_relation)

    if t.cuda.is_available():
        model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print(args.dataset + '-----------------------------------------')
    print('dim: %d' % args.dim, end='\t')
    print('K_u: %d' % args.K_u, end='\t')
    print('K_v: %d' % args.K_v, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
    train_auc_list = []
    train_acc_list = []
    eval_auc_list = []
    eval_acc_list = []
    test_auc_list = []
    test_acc_list = []
    HR_list = []
    NDCG_list = []

    for epoch in (range(args.epochs)):
        start = time.clock()
        loss_sum = 0
        np.random.shuffle(train_set)
        for i in range(0, len(train_set), args.batch_size):

            loss = model.cal_loss(train_set[i: i + args.batch_size], ripple_sets, user_records)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()

        train_auc, train_acc = eval_ctr(model, train_set, ripple_sets, user_records, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set, ripple_sets, user_records, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, ripple_sets, user_records, args.batch_size)

        print('epoch: %d \t train_auc: %.4f \t train_acc: %.4f \t '
              'eval_auc: %.4f \t eval_acc: %.4f \t test_auc: %.4f \t test_acc: %.4f \t' %
              ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        HR, NDCG = 0, 0
        if is_topk:
            HR, NDCG = eval_topk(model, rec, ripple_sets, user_records, args.batch_size, args.topk)
            print('HR: %.4f NDCG: %.4f' % (HR, NDCG), end='\t')

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        HR_list.append(HR)
        NDCG_list.append(NDCG)

        end = time.clock()
        print('time: %d' % (end - start))

    indices = eval_auc_list.index(max(eval_auc_list))
    print(args.dataset, end='\t')
    print('train_auc: %.4f \t train_acc: %.4f \t eval_auc: %.4f \t eval_acc: %.4f \t '
          'test_auc: %.4f \t test_acc: %.4f \t' %
          (train_auc_list[indices], train_acc_list[indices], eval_auc_list[indices], eval_acc_list[indices],
           test_auc_list[indices], test_acc_list[indices]), end='\t')

    print('HR: %.4f \t NDCG: %.4f' % (HR_list[indices], NDCG_list[indices]))

    return eval_auc_list[indices], eval_acc_list[indices], test_auc_list[indices], test_acc_list[indices]
