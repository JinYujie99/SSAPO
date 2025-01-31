import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import List, Optional
import datasets
from datasets import Dataset, DatasetDict
import multiprocessing

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

##### ATTENTION: modify 'alignment' to 'src.alignment', to redirect the import to src/alignment #####
from alignment import get_datasets, H4ArgumentParser
import datasets
import logging
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

datasets.disable_caching()

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(processName)s] %(message)s')
logger = logging.getLogger(__name__)

datasets.disable_caching()

@dataclass
class WstScriptArguments:
    """
    The arguments for the Worst Distribution Finding script.
    """
    input_dir: Optional[str] = field(
        default="datasets/",
        metadata={"help": "the location of the input training data"},
    )

    output_dir: Optional[str] = field(
        default="add-args",
        metadata={"help": "the location of the add-args training data"},
    )
    
    train_beta: Optional[float] = field(
        default=0.1,
        metadata={"help" : "training beta"}
    )
    
    wst_epsilon: Optional[float] = field(
        default=0.05,
        metadata={"help" : "max distance from the wst_distribution to the original data distribution"}
    ) 
    
    fit_K : Optional[int] = field(
        default=5,
        metadata={"help" : "number of linear functions to fit log function"}
    )


class OptimizationProblemSolver:
    def __init__(self, xi, K=5, epsilon=0.05, lambda_penalty=1e5):
        self.xi = xi
        self.N = len(xi)
        self.K = K
        self.epsilon = epsilon
        self.alpha = None
        self.q = None
        self.small_epsilon = 1e-15  # Small positive number to avoid division by zero
        self.lambda_penalty = lambda_penalty  # Penalty parameter for constraint violations

    def objective(self, x):
        alpha = x[:self.N * self.K].reshape(self.N, self.K)
        q = x[self.N * self.K:].reshape(self.N, self.K)
        ell_k = lambda k, v: -self.K / k * (v - k / self.K) - np.log(k / self.K)
        loss = 0
        penalty = 0
        
        for i in range(self.N):
            for k in range(self.K):
                if alpha[i, k] > self.small_epsilon:
                    loss += alpha[i, k] * ell_k(k + 1, self.xi[i] - q[i, k] / (alpha[i, k] + self.small_epsilon))
                    penalty += np.abs(np.sum(alpha[i]) - 1)
                    xi_minus_q_over_alpha = self.xi[i] - q[i, k] / (alpha[i, k] + self.small_epsilon)
                    penalty += max(0, -xi_minus_q_over_alpha)
                    penalty += max(0, xi_minus_q_over_alpha - 1)

        return -loss / self.N + self.lambda_penalty * penalty

    def constraint_sum_alpha_1(self, x, i):
        alpha = x[:self.N * self.K].reshape(self.N, self.K)
        return 1 - np.sum(alpha[i])

    def constraint_q_abs_sum(self, x):
        q = x[self.N * self.K:].reshape(self.N, self.K)
        return self.N * self.epsilon - np.sum(np.abs(q))

    def solve(self, id_begin):
        # Initial guess: all alpha_ik = 1/K, q_ik = 0
        initial_guess = np.zeros(2 * self.N * self.K)
        initial_guess[:self.N * self.K] = 1 / self.K
        
        constraints = []
        for i in range(self.N):
            constraints.append({'type': 'eq', 'fun': lambda x, i=i: self.constraint_sum_alpha_1(x, i)})
        
        constraints.append({'type': 'ineq', 'fun': self.constraint_q_abs_sum})
        
        bounds_alpha = [(1e-5, 1)] * self.N * self.K
        bounds_q = [(-self.epsilon, self.epsilon)] * self.N * self.K
        bounds = bounds_alpha + bounds_q
        
        options = {'maxiter': 50}
        initial_guess_obj = -self.objective(initial_guess)

        iteration_counter = 0  # Initialize the iteration counter
        def print_iteration(xk):
            nonlocal iteration_counter
            iteration_counter += 1
            if iteration_counter % 5 == 0:
                logger.info(f"Current iteration: {iteration_counter} with process id_begin : {id_begin}")

        result = minimize(self.objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, options=options, callback=print_iteration)
        
        logger.info(f"Optimized Objective with SLSQP : {-result.fun}, Initial Guess Objective : {initial_guess_obj}, Iterations : {iteration_counter}")
        
        if result.success:
            self.alpha = result.x[:self.N * self.K].reshape(self.N, self.K)
            self.q = result.x[self.N * self.K:].reshape(self.N, self.K)
            return self.alpha, self.q
        else:
            self.alpha = result.x[:self.N * self.K].reshape(self.N, self.K)
            self.q = result.x[self.N * self.K:].reshape(self.N, self.K)
            return self.alpha, self.q


# Parallel calculation function
def calculate(xi, id_begin, res_a, res_q):
    solver = OptimizationProblemSolver(xi, K, epsilon)
    alpha = list([list([1.0/K] * K)] * len(xi))
    q = list([list([0] * K)] * len(xi))

    try:
        # logger.info(id_begin)
        alpha, q = solver.solve(id_begin)
        for xii, qi, ai in zip(xi, q, alpha):
            if np.abs(1 - np.sum(ai)) > 1e-1:
                logger.info(f"ERROR : {xii}, {qi}, {ai}")
            for j in range(K):
                tmp = xii - qi[j] / ai[j]
                if tmp > 1.01 or tmp < -0.01:
                    logger.info(f"ERROR : {xii}, {qi[j]}, {ai[j]}")
    except ValueError as e:
        logger.info(e)

    # Store the result
    for i in range(len(xi)):
        res_a[id_begin + i] = alpha[i]
        res_q[id_begin + i] = q[i]


def main():
    parser = H4ArgumentParser((WstScriptArguments))
    wst_args = parser.parse_args()
    input_dir = wst_args.input_dir
    save_dir = wst_args.output_dir
    ori_datasets = datasets.load_from_disk(input_dir)
    global K, epsilon
    K = wst_args.fit_K
    epsilon = wst_args.wst_epsilon
    
    logger.info(f"wst_args: K {K}, epsilon {epsilon}")
    
    train_data = ori_datasets["train"]
    column_names = list(train_data.features)
    
    logger.info(f"column_names : {column_names}")
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    Xi = [sigmoid(0.1 * (item['chosen_reward'] - item['rejected_reward'])) for item in train_data]
    batch_size = 100
    
    cnt = 0
    tmp_data = []
    batches = []
    for item in Xi:
        cnt += 1
        tmp_data.append(item)
        if cnt == batch_size:
            batches.append(tmp_data)
            tmp_data = []
            cnt = 0
    if cnt > 0:
        batches.append(tmp_data)
        cnt = 0
    
    
    res_a = multiprocessing.Manager().list([list([1.0] * 1)] * len(train_data))
    res_q = multiprocessing.Manager().list([list([0.0] * 1)] * len(train_data))
    
    if epsilon > 1e-5:
        queue = multiprocessing.Queue()

        with multiprocessing.Pool(processes=128) as pool:
            args = [(batches[i], i * batch_size, res_a, res_q) for i in range(len(batches))]
            pool.starmap(calculate, args)
            
        while not queue.empty():
            logger.info(queue.get())
    
    final_data = []
    
    assert(len(train_data) == len(res_a))
    assert(len(res_a) == len(res_q))
    
    for item, ai, qi in zip(train_data, res_a, res_q):
        final_data.append({
            'index': item['index'],
            'prompt': item['prompt'],
            'chosen': item['chosen'],
            'rejected': item['rejected'],
            'chosen_reward': item['chosen_reward'],
            'rejected_reward': item['rejected_reward'],
            'ai' : ai,
            'qi' : [qx / ax for qx, ax in zip(qi, ai)]
        })
        
    new_data = Dataset.from_dict({
        'index': [item['index'] for item in final_data],
        'prompt': [item['prompt'] for item in final_data],
        'chosen': [item['chosen'] for item in final_data],
        'rejected': [item['rejected'] for item in final_data],
        'chosen_reward': [item['chosen_reward'] for item in final_data],
        'rejected_reward': [item['rejected_reward'] for item in final_data],
        'ai' : [item['ai'] for item in final_data],
        'qi' : [item['qi'] for item in final_data]
    })
    
    processed_dataset = DatasetDict({
        "train": new_data,
        "test": ori_datasets['test']
    })
    processed_dataset.save_to_disk(save_dir)


if __name__ == "__main__":
    main()