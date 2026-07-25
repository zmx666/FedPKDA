"""
NSL-KDD Traffic Detection Dataset Generator
Uses 12-process pool for data loading, preprocessing, and client partitioning.

Output: dataset/NSLKDD/train/{0..nc-1}.npz, dataset/NSLKDD/test/{0..nc-1}.npz, config.json
"""
import numpy as np
import os
import random
import warnings
import ujson
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
random.seed(1)
np.random.seed(1)

NUM_CLIENTS = 20
NUM_CLASSES = 2
TRAIN_SIZE = 0.75
ALPHA = 0.3
BATCH_SIZE = 16
N_WORKERS = 12
DIR_PATH = "NSLKDD/"

COLUMN_NAMES = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins',
    'logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate'
]
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
NUMERIC_COLS = [c for c in COLUMN_NAMES if c not in CATEGORICAL_COLS]


def load_csv_chunk(filepath, start, end):
    """Load a chunk of lines from the CSV."""
    rows, labels = [], []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            rows.append(parts[:41])
            label = parts[-2].strip().split('.')[0].strip('"').strip("'")
            labels.append(label)
    return rows, labels


def count_lines(filepath):
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)


def parallel_load_csv(filepath, n_workers=N_WORKERS):
    """Load CSV in parallel chunks."""
    n_lines = count_lines(filepath)
    chunk_size = (n_lines + n_workers - 1) // n_workers
    chunks = []
    for i in range(0, n_lines, chunk_size):
        end = min(i + chunk_size, n_lines)
        chunks.append((filepath, i, end))

    all_rows, all_labels = [], []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(load_csv_chunk, *c) for c in chunks]
        for f in as_completed(futures):
            rows, labels = f.result()
            all_rows.extend(rows)
            all_labels.extend(labels)
    return all_rows, all_labels


def preprocess_parallel(train_data, train_labels, test_data, test_labels):
    """Preprocess all data: encode categories, normalize numerics."""
    all_data = train_data + test_data
    n_train, n_test = len(train_data), len(test_data)
    all_labels = train_labels + test_labels
    n_total = len(all_data)

    cat_encoded = np.zeros((n_total, len(CATEGORICAL_COLS)), dtype=np.int64)
    for ci, col_name in enumerate(CATEGORICAL_COLS):
        idx = COLUMN_NAMES.index(col_name)
        values = [row[idx] for row in all_data]
        le = LabelEncoder()
        le.fit(values)
        cat_encoded[:, ci] = le.transform(values)

    # Bulk convert to numpy: pick numeric columns, handle missing values
    numeric_indices = [COLUMN_NAMES.index(c) for c in NUMERIC_COLS]
    all_array = np.array(all_data, dtype=str)
    num_data = np.where(all_array[:, numeric_indices] == '', '0.0',
                        all_array[:, numeric_indices]).astype(np.float64)

    scaler = StandardScaler()
    num_data = scaler.fit_transform(num_data)

    X_all = np.hstack([cat_encoded, num_data]).astype(np.float32)
    y_all = np.array([1 if l.lower() != 'normal' else 0 for l in all_labels], dtype=np.int64)

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]

    print(f"  X_train: {X_train.shape}, normal={np.sum(y_train==0)}, attack={np.sum(y_train==1)}")
    print(f"  X_test:  {X_test.shape}, normal={np.sum(y_test==0)}, attack={np.sum(y_test==1)}")
    return X_train, y_train, X_test, y_test


def separate_data(X, y, num_clients, alpha=0.3):
    """Dirichlet non-IID partition."""
    num_classes = len(np.unique(y))
    n_samples = len(y)
    min_size = 0
    least_samples = BATCH_SIZE / (1 - TRAIN_SIZE)

    while min_size < least_samples:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < n_samples / num_clients)
                                     for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    X_clients = [None] * num_clients
    y_clients = [None] * num_clients
    statistics = [None] * num_clients
    for cid in range(num_clients):
        idxs = idx_batch[cid]
        X_clients[cid] = X[idxs]
        y_clients[cid] = y[idxs]
        stats = [(int(l), int(np.sum(y_clients[cid] == l))) for l in np.unique(y_clients[cid])]
        statistics[cid] = stats
        print(f"  Client {cid}: {len(X_clients[cid])} samples, stats: {stats}")
    return X_clients, y_clients, statistics


def split_and_save_one(args):
    """Process one client: train/test split + save."""
    cid, X_c, y_c, train_path, test_path = args
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_c, y_c, train_size=TRAIN_SIZE, shuffle=True, random_state=cid
    )
    train_dict = {'x': X_tr.astype(np.float32), 'y': y_tr}
    test_dict = {'x': X_te.astype(np.float32), 'y': y_te}
    with open(train_path + str(cid) + '.npz', 'wb') as f:
        np.savez_compressed(f, data=train_dict)
    with open(test_path + str(cid) + '.npz', 'wb') as f:
        np.savez_compressed(f, data=test_dict)
    return len(y_tr), len(y_te)


def parallel_split_save(X_clients, y_clients, train_path, test_path):
    """Split and save all clients in parallel."""
    args_list = [(i, X_clients[i], y_clients[i], train_path, test_path) for i in range(len(y_clients))]
    total_train, total_test = 0, 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(split_and_save_one, args) for args in args_list]
        for f in as_completed(futures):
            tr, te = f.result()
            total_train += tr
            total_test += te
    print(f"  Total train: {total_train}, test: {total_test}")


def generate_nslkdd(dir_path, num_clients, num_classes):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    raw_dir = dir_path + "rawdata/"
    os.makedirs(raw_dir, exist_ok=True)

    train_file = raw_dir + "KDDTrain+.csv"
    test_file = raw_dir + "KDDTest+.csv"

    if not os.path.exists(train_file):
        print(f"ERROR: {train_file} not found. Download first:")
        print(f"  curl -o {train_file} https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt")
        return
    if not os.path.exists(test_file):
        print(f"ERROR: {test_file} not found. Download first:")
        print(f"  curl -o {test_file} https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt")
        return

    print(f"Loading train data from {train_file} ({N_WORKERS} workers)...")
    train_data_raw, train_labels_raw = parallel_load_csv(train_file)
    print(f"  {len(train_data_raw)} samples loaded.")

    print(f"Loading test data from {test_file} ({N_WORKERS} workers)...")
    test_data_raw, test_labels_raw = parallel_load_csv(test_file)
    print(f"  {len(test_data_raw)} samples loaded.")

    print("Preprocessing...")
    X_train, y_train, X_test, y_test = preprocess_parallel(
        train_data_raw, train_labels_raw, test_data_raw, test_labels_raw
    )

    print("Combining and partitioning across clients (Dirichlet)...")
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    X_clients, y_clients, statistics = separate_data(X_all, y_all, num_clients, alpha=ALPHA)

    print(f"Saving client data (parallel, {N_WORKERS} workers)...")
    parallel_split_save(X_clients, y_clients, train_path, test_path)

    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': True,
        'balance': True,
        'partition': 'dir',
        'Size of samples for labels in clients': statistics,
        'alpha': ALPHA,
        'batch_size': BATCH_SIZE,
    }
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("NSL-KDD dataset generation complete!\n")


if __name__ == "__main__":
    generate_nslkdd(DIR_PATH, NUM_CLIENTS, NUM_CLASSES)
