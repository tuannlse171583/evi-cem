import torch
import pytorch_lightning as pl
from utils import compute_metric

from torchvision.models import resnet34, ResNet34_Weights

class CBM(pl.LightningModule):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        embedding_activation="leakyrelu",
        c_extractor_arch="resnet34",
        optimizer="adam",
        learning_rate=0.01,
        weight_decay=4e-05,
        momentum=0.9,
        train_with_c_gt=False,
        concept_weight=None,
        interven_prob=0.25,
        concept_loss_weight=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.emb_size = emb_size

        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.train_with_c_gt = train_with_c_gt
        self.concept_weight = concept_weight.cuda() if concept_weight is not None else None # Handle case where concept_weight is None
        self.interven_prob = interven_prob
        self.concept_loss_weight = concept_loss_weight

        if c_extractor_arch == "resnet34":
            self.pre_concept_model = resnet34(weights=ResNet34_Weights.DEFAULT)
            backbone_embed_size = list(self.pre_concept_model.modules())[-1].in_features
            self.pre_concept_model.fc = torch.nn.Identity()
        else:
            raise NotImplementedError

        if embedding_activation == "sigmoid":
            embed_act = torch.nn.Sigmoid()
        elif embedding_activation == "leakyrelu":
            embed_act = torch.nn.LeakyReLU()
        elif embedding_activation == "relu":
            embed_act = torch.nn.ReLU()
        elif embedding_activation is None:
            embed_act = torch.nn.Identity()

        self.concept_context_generators = torch.nn.ModuleList()
        # Replaces alpha and beta generators with a single linear layer
        self.concept_prob_generator = torch.nn.Linear(2 * emb_size, 1)
        torch.nn.init.xavier_uniform_(self.concept_prob_generator.weight)
        torch.nn.init.constant_(self.concept_prob_generator.bias, 0.0)

        for _ in range(n_concepts):
            concept_context_generator = torch.nn.Sequential(
                torch.nn.Linear(backbone_embed_size, 2 * emb_size), embed_act
            )
            torch.nn.init.xavier_uniform_(concept_context_generator[0].weight)
            torch.nn.init.constant_(concept_context_generator[0].bias, 0.0)
            self.concept_context_generators.append(concept_context_generator)

        self.c2y_model = torch.nn.Sequential(*[torch.nn.Linear(n_concepts * emb_size, n_tasks)])
        self.loss_task = torch.nn.CrossEntropyLoss()
        # Define concept loss
        self.loss_concept = torch.nn.BCELoss(weight=concept_weight) if concept_weight is not None else torch.nn.BCELoss()

    def _after_interventions(self, prob, c_true):
        # we will probabilistically intervene in some concepts
        mask = torch.bernoulli(torch.ones(self.n_concepts) * self.interven_prob)
        interven_idxs = torch.tile(mask, (c_true.shape[0], 1)).float().to(prob.device)
        return prob * (1 - interven_idxs) + interven_idxs * c_true

    def forward(self, x, c, train=False):
        pre_c = self.pre_concept_model(x)
        contexts, c_probs = [], []

        for context_gen in self.concept_context_generators:
            context = context_gen(pre_c)
            contexts.append(context)
            c_probs.append(torch.sigmoid(self.concept_prob_generator(context)))

        c_probs = torch.cat(c_probs, dim=-1)  # Concatenate concept probabilities
        contexts = torch.stack(contexts, dim=1)

        if train and self.interven_prob != 0:
            c_hard = torch.where(c > 0.5, 1.0, 0.0)
            c_probs_mix = self._after_interventions(c_probs, c_true=c_hard)
        else:
            c_probs_mix = c_probs

        contexts_pos = contexts[:, :, : self.emb_size]
        contexts_neg = contexts[:, :, self.emb_size :]
        c_pred = contexts_pos * c_probs_mix.unsqueeze(-1) + contexts_neg * (1 - c_probs_mix.unsqueeze(-1))
        c_pred = c_pred.view((-1, self.emb_size * self.n_concepts))

        y = self.c2y_model(c_pred)

        return c_probs, y

    def _run_step(self, batch, train):
        x, y, c, soft_c, _ = batch
        c_probs, y_logits = self.forward(x, c, train)

        task_loss = self.loss_task(y_logits, y)
        task_loss_scalar = task_loss.detach()

        if self.concept_loss_weight != 0:
            # Determine the target concept labels for the loss
            target_c = c if self.train_with_c_gt else soft_c
            concept_loss = self.loss_concept(c_probs, target_c)
            concept_loss_scalar = concept_loss.detach()
            loss = self.concept_loss_weight * concept_loss + task_loss
        else:
            loss = task_loss
            concept_loss_scalar = torch.tensor(0.0) # Ensuring a scalar tensor is used

        (c_acc, c_auc, c_f1), (y_acc, y_auc, y_f1) = compute_metric(c_probs, y_logits, c, y)
        result = {
            "c_acc": c_acc,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_acc": y_acc,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "concept_loss": concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "avg_c_y_acc": (c_acc + y_acc) / 2,
            "avg_c_y_auc": (c_auc + y_auc) / 2,
            "avg_c_y_f1": (c_f1 + y_f1) / 2,
        }
        return loss, result

    def training_step(self, batch, batch_idx):
        loss, result = self._run_step(batch, train=True)
        for name, val in result.items():
            if "loss" in name:
                self.log(f"train/loss/{name}", val)
            else:
                self.log(f"train/metric/{name}", val)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        _, result = self._run_step(batch, train=False)
        for name, val in result.items():
            if "loss" in name:
                self.log(f"val/loss/{name}", val)
            else:
                self.log(f"val/metric/{name}", val)
        result = {"val_" + key: val for key, val in result.items()}
        return result

    def test_step(self, batch, batch_idx):
        _, result = self._run_step(batch, train=False)
        for name, val in result.items():
            self.log(f"test/{name}", val, prog_bar=True)
        return result["loss"]

    def predict_step(self, batch, batch_idx):
        x, _, c, _, _ = batch
        c_probs, y_logits = self.forward(x, c, train=False)  # Fixed: Added c to forward call
        return c_probs, y_logits #Trả về 1 tuple gồm c_probs and y_logits

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train/loss/loss",
        }
