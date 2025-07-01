# Created by
import numpy as np
import time
from solution import solution
import random


def beta_group_pos(num_alpha, num_beta, scores, pos):
    top_large_idx = np.argsort(-scores)[:num_alpha + num_beta]
    beta_idx = top_large_idx[num_alpha:]
    return pos[beta_idx], scores[beta_idx]


def delta_group_pos(num_alpha, num_beta, num_delta, scores, pos):
    top_large_idx = np.argsort(-scores)[:num_alpha + num_beta + num_delta]
    delta_idx = top_large_idx[num_alpha + num_beta:]
    return pos[delta_idx], scores[delta_idx]


def record_best(pos, scores, num_alpha, num_beta, num_delta):
    alpha_score = max(scores)
    alpha = pos[np.argmax(scores)]

    betas, beta_scores = beta_group_pos(num_alpha, num_beta, scores, pos)
    deltas, delta_scores = delta_group_pos(num_alpha, num_beta, num_delta, scores, pos)
    return alpha, alpha_score, betas, beta_scores, deltas, delta_scores


def random_selection(alpha, betas, deltas, ):
    r = np.random.random()
    if r <= 0.2:
        offspring = deltas[np.random.choice(np.arange(len(deltas)))]
    elif r <= 0.5:
        offspring = betas[np.random.choice(np.arange(len(betas)))]
    else:
        offspring = alpha
    return offspring


def k_distrb(x, idx, lb=0, ub=255, k=1, radius=5):
    unmask = np.random.choice(np.arange(len(idx)), size=k, replace=False)
    rand_pos = np.random.randint(low=-radius, high=radius + 1, size=k)
    update = np.clip(idx + rand_pos, lb, ub)
    change_idx = np.where(idx != update)[0]
    new_pos = x.copy()
    new_pos[change_idx] = 0
    new_pos[update[change_idx]] = 1
    return new_pos


def roulette_wheel_selection(cum_probs):
    for c, cumulative_prob in enumerate(cum_probs):
        if np.random.random() <= cumulative_prob:
            return c


def pair_roulette_selection(scores):
    cum_prob = np.cumsum(scores / np.sum(scores))
    parent1_id = roulette_wheel_selection(cum_prob)
    parent2_id = parent1_id
    while parent2_id == parent1_id:
        parent2_id = roulette_wheel_selection(cum_prob)
    return parent1_id, parent2_id


def map_one_crossover(parent1, parent2, card):
    map1 = np.where(parent1 == 1)[0]
    map2 = np.where(parent2 == 1)[0]
    offspring = np.zeros_like(parent1)
    p = []
    for i in range(card):
        selected_map = random.choice([map1, map2])
        if selected_map.size == 0:
            selected_map = map2 if selected_map is map1 else map1
        q = np.random.choice(selected_map)
        p.append(q)
        map1 = map1[map1 != q]
        map2 = map2[map2 != q]
    offspring[np.array(p)] = 1
    return offspring


def mutate(x):
    zeros_indices = np.where(x == 0)[0]
    ones_indices = np.where(x == 1)[0]
    zero_to_flip_index = zeros_indices[np.random.choice(len(zeros_indices), replace=False)]
    one_to_flip_index = ones_indices[np.random.choice(len(ones_indices), replace=False)]
    x[zero_to_flip_index] = 1
    x[one_to_flip_index] = 0
    return x


def update_best_drop_replicate(old, new):
    a, a_sc, b, b_sc, d, d_sc = old
    na, na_sc, nb, nb_sc, nd, nd_sc = new
    if len(a.shape) != len(b.shape):
        a = a.reshape(-1, a.shape[0])
        na = na.reshape(-1, na.shape[0])
        a_sc = np.array([a_sc])
        na_sc = np.array([na_sc])
    lead_pos = np.concatenate((a, b, d, na, nb, nd))
    lead_scores = np.concatenate((a_sc, b_sc, d_sc, na_sc, nb_sc, nd_sc))
    keep_idx = np.argsort(-lead_scores)[:round(len(lead_pos) / 2)]
    update_pos = lead_pos[keep_idx]
    update_scores = lead_scores[keep_idx]
    return update_pos, update_scores


def decoder(x, value_set):
    th = [value_set[index] for index, value in enumerate(x) if value == 1]
    return np.array(th)


class DGWO:
    def __init__(self, objf, lb, ub, Agents_no, Max_iter, num_alpha=1, num_beta=2, num_delta=3, pf=0.9, pm=0.01):
        self.objf = objf  # optimization criterion
        self.lb = lb
        self.ub = ub
        self.SearchAgents_no = Agents_no
        self.Max_iter = Max_iter
        self.num_alpha = num_alpha
        self.num_beta = num_beta
        self.num_delta = num_delta
        self.pf = pf
        self.pm = pm

    def process(self, card, dim=256):
        # card: cardinality of non-zero element set
        # dim: in rgb images, grayscale intensity value ranges from 0 to 255
        s = solution()
        # initialization
        non_zero_idx = np.array(
            [np.random.choice(dim - 1, size=card, replace=False) for i in range(self.SearchAgents_no)])
        positions = np.zeros((self.SearchAgents_no, dim))
        for i in range(self.SearchAgents_no):
            positions[i, non_zero_idx[i]] = 1
        scores = np.array([self.objf(positions[i], card) for i in range(self.SearchAgents_no)])

        convergence = []
        value_set = np.arange(dim)
        print("DGWO starts...")
        st = time.time()

        # determine the leading wolf community
        alpha, alpha_score, betas, beta_scores, deltas, delta_scores = record_best(positions, scores, self.num_alpha,
                                                                                   self.num_beta, self.num_delta)

        for t in range(self.Max_iter):
            new_pos = np.zeros_like(positions)
            new_scores = np.zeros_like(scores)
            i = 0  # counter
            while i < self.SearchAgents_no:
                if np.random.random() <= self.pf:
                    offspring = random_selection(alpha, betas, deltas)
                    idx = np.where(offspring == 1)[0]
                    new_pos[i] = k_distrb(offspring, idx, self.lb, self.ub)
                else:
                    parent1_idx, parent2_idx = pair_roulette_selection(scores)
                    parent1, parent2 = positions[parent1_idx], positions[parent2_idx]
                    offspring = map_one_crossover(parent1, parent2, card)
                # mutate
                if np.random.random() <= self.pm:
                    offspring = mutate(offspring)
                new_pos[i] = offspring
                new_scores[i] = self.objf(new_pos[i], card)
                i += 1
            new_a, new_a_sc, new_b, new_b_sc, new_d, new_d_sc = record_best(new_pos, new_scores, self.num_alpha,
                                                                            self.num_beta, self.num_delta)
            old = (alpha, alpha_score, betas, beta_scores, deltas, delta_scores)
            new = (new_a, new_a_sc, new_b, new_b_sc, new_d, new_d_sc)
            update_pos, update_scores = update_best_drop_replicate(old, new)
            alpha, betas, deltas = update_pos[:self.num_alpha].squeeze(), update_pos[
                                                                          self.num_alpha:self.num_alpha + self.num_beta], update_pos[
                                                                                                                          self.num_alpha + self.num_beta:]
            alpha_score, beta_scores, delta_scores = update_scores[:self.num_alpha].squeeze(), update_scores[
                                                                                               self.num_alpha:self.num_alpha + self.num_beta], update_scores[
                                                                                                                                               self.num_alpha + self.num_beta:]
            # update all agents' scores and positions
            positions = new_pos
            scores = new_scores

            convergence.append(alpha_score)
        et = time.time()
        print("DGWO ends...")

        s.executionTime = et - st
        s.convergence = np.array(convergence)
        s.best = alpha_score
        s.bestIndividual = decoder(alpha, value_set)

        return s
