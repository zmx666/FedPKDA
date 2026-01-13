import time
import copy
import numpy as np
from ..clients.clientpkda import clientpkda
from ..servers.serverbase import Server
from sklearn.cluster import KMeans
import torch

class Fedpkda(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientpkda)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        # self.load_model()
        self.Budget = []
        self.global_par= args.global_par
        self.local_par=args.local_par

    def all_clients(self):
        return self.clients

    def aggregate_wrt_fisher(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        FIM_weight_list = []
        for id in self.uploaded_ids:
            FIM_weight_list.append(self.clients[id].fim_trace_history[-1])
        # normalization to obtain weight
        FIM_weight_list = [FIM_value/sum(FIM_weight_list) for FIM_value in FIM_weight_list]

        for w, client_model in zip(FIM_weight_list, self.uploaded_models):
            self.add_parameters(w, client_model)
    def send_selected_models(self, selected_ids, epoch):
        assert len(self.clients) > 0
        self.global_prototypes = self.compute_global_prototypes()
        for client in [c for c in self.clients if c.id in selected_ids]:
            start_time = time.time()
            self.progress = epoch / self.global_rounds
            client.set_parameters(self.global_model, self.global_prototypes, self.progress,epoch)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    def compute_global_prototypes(self):
        client_prototypes = [client.get_noisy_local_prototypes() for client in self.clients] # 20 clients
        num_classes = len(client_prototypes[0])
        global_prototypes = [None] * num_classes
        for k in range(num_classes):
            class_prototypes = [proto[k] for proto in client_prototypes if proto[k] is not None]
            if len(class_prototypes) >= 3:
                data = torch.stack(class_prototypes).cpu().numpy()
                #KMeans
                kmeans = KMeans(n_clusters=1, random_state=42).fit(data)
                cluster_center = kmeans.cluster_centers_[0]
                cov = np.cov(data.T) + np.eye(data.shape[1]) * 1e-6
                inv_cov = np.linalg.inv(cov)
                distances = np.array([np.sqrt((p - cluster_center) @ inv_cov @ (p - cluster_center).T) for p in data])
                weights = 1 / (distances + 1e-6)
                weights = weights / weights.sum()
                weighted_proto = np.sum(weights[:, None] * data, axis=0)
                global_prototypes[k] = torch.from_numpy(weighted_proto).to(self.device).float()
            else:
                global_prototypes[k] = None
        return global_prototypes
#train
    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.alled_clients = self.all_clients()

            selected_ids = [client.id for client in self.selected_clients]

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            self.send_selected_models(selected_ids, i)
            for client in self.alled_clients:
                # print("===============")
                client.train(client.id in selected_ids)
            self.receive_models()
            self.aggregate_wrt_fisher()
            # self.aggregate_parameters()
            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        print(f'+++++++++++++++++++++++++++++++++++++++++')
        gen_acc = self.avg_generalization_metrics()
        print(f'Generalization Acc: {gen_acc}')
        print(f'+++++++++++++++++++++++++++++++++++++++++')
        self.save_results()
        self.save_global_model()

