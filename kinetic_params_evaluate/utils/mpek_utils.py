import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import r2_score


class Expert(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, expert_hid_layer=3, dropout_rate=0.3):
        super(Expert, self).__init__()
        self.expert = MLP(
            in_size=in_size,
            hidden_size=hidden_size,
            out_size=out_size,
            layer_num=expert_hid_layer,
            dropout_rate=dropout_rate
        )

    def forward(self, x):
        return self.expert(x)


class GateModule(nn.Module):
    def __init__(self, gate_in, num_experts, num_tasks):
        super(GateModule, self).__init__()

        self.share_gate = Gate(gate_in, num_experts * (num_tasks + 1))
        self.task_gates = nn.ModuleList(
            [Gate(gate_in, num_experts * 2) for _ in range(num_tasks)]
        )
        self.gate_activation = nn.Softmax(dim=-1)

    def forward(self, share_x, task_x):
        assert len(task_x) == len(self.task_gates)

        share_gate_out = self.share_gate(share_x)
        share_gate_out = self.gate_activation(share_gate_out)

        task_gate_out_list = [e(task_x[i]) for i, e in enumerate(self.task_gates)]

        return share_gate_out, task_gate_out_list


class Gate(nn.Module):
    def __init__(self, input_size, output_size):
        super(Gate, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc1(x)


class RBF(nn.Module):
    def __init__(self, centers, gamma, device):
        super(RBF, self).__init__()
        self.centers = torch.reshape(torch.tensor(centers, dtype=torch.float32), [1, -1]).to(device)
        self.gamma = gamma
        self.device = device

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = torch.reshape(x, [-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers)).to(self.device)


class ConditionFloatRBF(nn.Module):
    def __init__(self, embed_dim, device):
        super(ConditionFloatRBF, self).__init__()

        self.condition_names = ["pH", "temperature"]
        self.rbf_params = {
            'pH': (np.arange(0, 14, 0.1), 10.0),
            'temperature': (np.arange(0, 100, 1), 10.0),
            'logp': (np.arange(-20, 20.1, 0.1), 5.0),
            'mw': (np.arange(1, 1101, 1), 0.1)
        }

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()

        for name in self.condition_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma, device=device)

            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim)
            self.linear_list.append(linear)

    def forward(self, condition):
        out_embed = 0
        for i, name in enumerate(self.condition_names):
            if name in condition:
                x = condition[name]
                if x is not None:
                    rbf_x = self.rbf_list[i](x)
                    out_embed += self.linear_list[i](rbf_x)
        return out_embed


class MPEKModel(nn.Module):
    def __init__(
            self,
            expert_out: int, expert_hidden: int, expert_layers: int, num_experts: int,
            num_tasks: int, tower_hid_layer=5, tower_hid_unit=128, weights=[1, 1], tower_dropout=[0.2, 0.2],
            dropout=0.2, num_ple_layers=1, gen_vec=False, device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.gen_vec = gen_vec

        ligand_enc_embed_dim = 300
        self.pro2lig = nn.Linear(1024, ligand_enc_embed_dim)
        self.h_aux_encoder = ConditionFloatRBF(embed_dim=ligand_enc_embed_dim, device=device)
        self.organism2lig = nn.Linear(8, ligand_enc_embed_dim)

        input_size = ligand_enc_embed_dim * 3
        self.weights = weights

        self.multi_block = PLE(
            experts_in=input_size,
            experts_out=expert_out,
            experts_hidden=expert_hidden,
            expert_hid_layer=expert_layers,
            dropout_rate=dropout,
            num_experts=num_experts,
            num_tasks=num_tasks,
            num_ple_layers=num_ple_layers
        )

        if not self.gen_vec:
            self.towers = nn.ModuleList(
                [
                    MLP(
                        in_size=expert_out,
                        hidden_size=tower_hid_unit,
                        out_size=1,
                        layer_num=tower_hid_layer,
                        dropout_rate=tower_dropout[i]
                    ) for i in range(num_tasks)
                ]
            )

        self.criterion = nn.MSELoss()

    def forward(self, x, task_names=['km', 'kcat']):
        x = x.float()
        h_lig, h_prot, h_aux = x[:, 1024:1024+300], x[:, :1024], x[:, 1024+300:]  # TODO
        h_prot = self.pro2lig(h_prot)

        # temperature  # TODO 顺序
        condition = {'pH': h_aux[:, 0], 'temperature': h_aux[:, 1], 'mw': h_aux[:, 2], 'logp': h_aux[:, 3]}
        h_aux_embedding = self.h_aux_encoder(condition)

        # # organism
        # h_organism = self.organism2lig(h_aux[:, -8:])
        # h_aux_embedding = h_aux_embedding + h_organism

        h = torch.cat([h_lig, h_prot, h_aux_embedding], 1)  # [bs, 512 * 3]

        tower_input = self.multi_block(h)

        if self.gen_vec:
            return np.hstack([x.cpu().numpy() for x in tower_input])

        pred = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return dict(zip(task_names, pred))

    def compute_loss(self, net_outputs, labels):
        weights_mapping = dict(zip(labels, self.weights))
        loss_dict = {}
        loss = 0

        for task_name, target in labels.items():
            predict = net_outputs[task_name]
            mask = ~torch.isnan(target)
            target_valid = target[mask]
            predict_valid = predict[mask]

            if target_valid.numel() == 0:
                mse = torch.tensor(0.0, device=target.device)  # 如果没有有效值，则损失为0
            else:
                mse = torch.mean((predict_valid - target_valid) ** 2.0)

            loss += mse * weights_mapping[task_name]
            loss_dict[f"{task_name}_loss"] = loss_dict.get(f"{task_name}_loss", 0) + mse.detach().item()

        loss_dict["loss"] = loss.detach().item()

        return loss, loss_dict


    def compute_r2(self, net_outputs, labels):
        r2_dict = {}
        for task_name, target in labels.items():
            predict = net_outputs[task_name].flatten()
            mask = ~torch.isnan(target)
            predict = predict[mask].cpu().numpy()
            target = target[mask].cpu().numpy()

            # 计算 R² 分数
            tmp_r2 = r2_score(target, predict)  # 注意顺序，先 target 再 predict
            r2_dict[f"{task_name}_r2"] = tmp_r2

        return r2_dict


class MLP(nn.Module):
    def __init__(self, layer_num, in_size, hidden_size, out_size, dropout_rate):
        super(MLP, self).__init__()

        layers = []

        if layer_num == 1:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.ReLU())

        else:
            for layer_id in range(layer_num):
                if layer_id == 0:
                    layers.append(nn.Linear(in_size, hidden_size))
                    layers.append(nn.Dropout(dropout_rate))
                    layers.append(nn.ReLU())
                elif layer_id < layer_num - 1:
                    layers.append(nn.Linear(hidden_size, hidden_size))
                    layers.append(nn.Dropout(dropout_rate))
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Linear(hidden_size, out_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PLE(nn.Module):
    def __init__(self, experts_in, experts_out,
                 experts_hidden, expert_hid_layer=3,
                 dropout_rate=0.3, num_experts=5, num_tasks=2, num_ple_layers=1):
        super(PLE, self).__init__()

        self.layers = nn.ModuleList([])
        self.num_tasks = num_tasks

        for i in range(num_ple_layers):
            if i == 0:
                layer = PleLayer(
                    experts_in, experts_out, experts_hidden,
                    expert_hid_layer, dropout_rate,
                    num_experts, num_tasks
                )
            else:
                layer = PleLayer(
                    experts_out, experts_out, experts_hidden,
                    expert_hid_layer, dropout_rate,
                    num_experts, num_tasks
                )

            self.layers.append(layer)

    def forward(self, x):
        share_x, task_x = x, [x for _ in range(self.num_tasks)]

        for layer in self.layers:
            share_x, task_x = layer(share_x, task_x)

        return task_x


class PleLayer(nn.Module):
    def __init__(
            self,
            experts_in, experts_out, experts_hidden,
            expert_hid_layer=3, dropout_rate=0.3,
            num_experts=5, num_tasks=2
    ):
        super(PleLayer, self).__init__()

        self.experts = ExpertModule(
            experts_in, experts_out, experts_hidden, expert_hid_layer, dropout_rate, num_experts, num_tasks
        )

        self.gates = GateModule(experts_in, num_experts, num_tasks)

    def forward(self, share_x, task_x):
        share_expert_out, task_expert_out_list = self.experts(share_x, task_x)
        share_gate_out, task_gate_out_list = self.gates(share_x, task_x)

        task_out_list = []
        for i in range(len(task_x)):
            task_expert_out = task_expert_out_list[i]
            task_gate_out = task_gate_out_list[i]

            task_out = torch.cat([share_expert_out, task_expert_out], dim=1)
            task_out = torch.einsum('be,beu -> beu', task_gate_out, task_out)
            task_out = task_out.sum(dim=1)

            task_out_list.append(task_out)

        share_out = torch.cat([share_expert_out, *task_expert_out_list], dim=1)
        share_out = torch.einsum('be,beu -> beu', share_gate_out, share_out)
        share_out = share_out.sum(dim=1)

        return share_out, task_out_list


class ExpertModule(nn.Module):
    def __init__(self, experts_in, experts_out, experts_hidden, expert_hid_layer=3, dropout_rate=0.3, num_experts=5,
                 num_tasks=2):
        super(ExpertModule, self).__init__()

        self.num_experts = num_experts
        self.experts_out = experts_out

        self.share_expert = nn.ModuleList(
            [
                Expert(
                    in_size=experts_in,
                    out_size=experts_out,
                    hidden_size=experts_hidden,
                    expert_hid_layer=expert_hid_layer,
                    dropout_rate=dropout_rate
                ) for _ in range(num_experts)
            ]
        )

        self.task_experts = nn.ModuleList([])
        for _ in range(num_tasks):
            task_expert = nn.ModuleList(
                [
                    Expert(
                        in_size=experts_in,
                        out_size=experts_out,
                        hidden_size=experts_hidden,
                        expert_hid_layer=expert_hid_layer,
                        dropout_rate=dropout_rate
                    ) for _ in range(num_experts)
                ]
            )
            self.task_experts.append(task_expert)

        # self.expert_activation = nn.ReLU()

    def forward(self, share_x, task_x):
        assert len(task_x) == len(self.task_experts)

        share_expert_out = [e(share_x) for e in self.share_expert]
        share_expert_out = torch.concat(share_expert_out, dim=0).view(-1, self.num_experts, self.experts_out)
        # share_expert_out = self.expert_activation(share_expert_out)

        task_expert_out_list = []
        for i, task_expert in enumerate(self.task_experts):
            task_expert_out = [e(task_x[i]) for e in task_expert]
            task_expert_out = torch.concat(task_expert_out, dim=0).view(-1, self.num_experts, self.experts_out)
            # task_expert_out = self.expert_activation(task_expert_out)
            task_expert_out_list.append(task_expert_out)

        return share_expert_out, task_expert_out_list


class AuxiliaryEncoder(nn.Module):
    def __init__(self, organism_dict_size: int, embed_size: int):
        super().__init__()
        self.embed_dim = embed_size

        self.organism_embedding = nn.Embedding(organism_dict_size + 1, embed_size)
        self.cond_embedding = ConditionFloatRBF(embed_size)

    def forward(self, organism, condition=None):
        out = 0
        out += self.organism_embedding(organism)

        out += self.cond_embedding(condition)

        return out


# if __name__ == '__main__':
#     device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
#     model = MTLModel(expert_out=768, expert_hidden=768,
#                      expert_layers=1, num_experts=4, num_tasks=2).to(device)
#     task_names = {'km', 'kcat'}
#     # 使用 torchsummary 打印模型结构，输入形状为 (6, 1088)
#     # summary(model, input_size=(6, 1088))
#     random_tensor = torch.randn(6, 1088).to(device)
#     y = model(random_tensor)
#     print(y)