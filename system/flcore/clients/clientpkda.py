import numpy as np
import time
import copy
from ..clients.clientbase import Client
from torch.autograd import grad
import  torch
from ..utils.func import Func
class clientpkda(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.fim_trace_history = []
        self.model_cache=[]
        #init
        self.noise_scale = 0.1
        self.learn_scale = torch.nn.Parameter(torch.tensor(5.0, requires_grad=True))
        self.local_par = getattr(args, "local_par", 1.0)
        self.global_par = getattr(args, "global_par", 1.0)
        self.func=Func()

    def train(self, is_selected):
        if is_selected:
            trainloader = self.load_train_data()
            # self.model.to(self.device)
            self.model.train()
            start_time = time.time()
            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            for step in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
            self.model.eval()
            fim_trace_sum = 0
            for i, (x, y) in enumerate(self.load_train_data()):
                # Forward pass
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()
                grads = grad(nll, self.model.parameters())
                #  Fisher
                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()
            self.fim_trace_history.append(fim_trace_sum.item())

        else:
            trainloader = self.load_train_data()
            self.model.eval()
            fim_trace_sum = 0
            for i, (x, y) in enumerate(trainloader):
                # Forward pass
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()
                grads = grad(nll, self.model.parameters())
                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()
            # add the fisher log
            self.fim_trace_history.append(fim_trace_sum.item())

    def evaluate(self):
        testloader = self.load_test_data()
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        accuracy = 100. * correct / total
        return accuracy
#noise
    def get_noisy_local_prototypes(self):
        local_prototypes = [[] for _ in range(self.num_classes)]
        trainloader = self.load_train_data()
        for x_batch, y_batch in trainloader:
            if type(x_batch) == type([]):
                x_batch[0] = x_batch[0].to(self.device)
            else:
                x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            with torch.no_grad():
                proto_batch = self.model.base(x_batch)
            proto_batch_one = torch.clamp(proto_batch, -1, 1)
            for proto, y in zip(proto_batch_one, y_batch):
                local_prototypes[y.item()].append(proto)
        mean_prototypes = []
        for class_prototypes in local_prototypes:
            if class_prototypes:
                stacked_protos = torch.stack(class_prototypes)
                mean_proto = torch.mean(stacked_protos, dim=0)
                laplace_distribution = torch.distributions.Laplace(loc=0, scale=self.noise_scale)
                noisy_proto = mean_proto + laplace_distribution.sample(mean_proto.shape).to(self.device)
                mean_prototypes.append(noisy_proto)
            else:
                mean_prototypes.append(None)
        return mean_prototypes
#local
    def get_local_prototypes(self):
        local_prototypes = [[] for _ in range(self.num_classes)]
        trainloader = self.load_train_data()
        for x_batch, y_batch in trainloader:
            if type(x_batch) == type([]):
                x_batch[0] = x_batch[0].to(self.device)
            else:
                x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            with torch.no_grad():
                proto_batch = self.model.base(x_batch)
            for proto, y in zip(proto_batch, y_batch):
                local_prototypes[y.item()].append(proto)
        mean_prototypes = []
        for class_prototypes in local_prototypes:
            if class_prototypes:
                stacked_protos = torch.stack(class_prototypes)
                mean_proto = torch.mean(stacked_protos, dim=0)
                mean_prototypes.append(mean_proto)
            else:
                mean_prototypes.append(None)
        return mean_prototypes

    def set_parameters(self, model, global_prototypes, progress, epoch):
        #init
        lambda_PA = torch.tensor(0.1, dtype=torch.float32, device=self.device)
        lambda_FA = torch.tensor(0.1, dtype=torch.float32, device=self.device)
        local_prototypes = self.get_local_prototypes()
        self.model.load_state_dict(model.state_dict())
        trainloader = self.load_train_data()
        #optimizer
        alignment_optimizer = torch.optim.SGD(
            list(model.base.parameters()) + [self.learn_scale], lr=0.002
        )
        alignment_contrastive_loss_fn = torch.nn.MSELoss()
        with torch.no_grad():
            progress = torch.tensor(progress, dtype=torch.float32)
            activation_value = self.func(progress * self.learn_scale)
            lambda_PA += self.local_par * (1 - activation_value)
            lambda_FA += self.global_par * activation_value
        for _ in range(1):
            for x_batch, y_batch in trainloader:
                if type(x_batch) == type([]):
                    x_batch[0] = x_batch[0].to(self.device)
                else:
                    x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                global_proto_batch = model.base(x_batch)
                loss = 0.0
                unique_labels = y_batch.unique()
                for label in unique_labels:
                    label_idx = label.item()
                    class_mask = (y_batch == label)
                    if local_prototypes[label_idx] is not None and class_mask.any():
                        anchor = global_proto_batch[class_mask]
                        if anchor.shape[0] > 0:
                            loss += lambda_PA*alignment_contrastive_loss_fn(
                                anchor, local_prototypes[label_idx].unsqueeze(0).expand_as(anchor)
                            )
                    if global_prototypes and global_prototypes[label_idx] is not None and class_mask.any():
                        anchor = global_proto_batch[class_mask]
                        if anchor.shape[0] > 0:
                            loss += lambda_FA*alignment_contrastive_loss_fn(
                                anchor, global_prototypes[label_idx].unsqueeze(0).expand_as(anchor)
                            )
                alignment_optimizer.zero_grad()
                loss.backward()
                alignment_optimizer.step()
        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data.copy_(new_param.data)
