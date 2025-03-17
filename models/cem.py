import torch
from utils import compute_metric
from models.clm import ConceptLearningModel


class CEM(ConceptLearningModel):
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
        super().__init__(
            n_concepts,
            emb_size,
            embedding_activation,
            c_extractor_arch,
            optimizer,
            learning_rate,
            weight_decay,
            momentum,
            train_with_c_gt,
            concept_weight,
        )
        self.interven_prob = interven_prob
        self.concept_loss_weight = concept_loss_weight
        self.c2y_model = torch.nn.Sequential(*[torch.nn.Linear(n_concepts, n_tasks)])
        self.loss_task = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def _after_interventions(self, prob, c_true):
        mask = torch.bernoulli(torch.ones(self.n_concepts) * self.interven_prob)
        interven_idxs = torch.tile(mask, (c_true.shape[0], 1)).float().to(prob.device)
        return prob * (1 - interven_idxs) + interven_idxs * c_true

    def forward(self, x, c, train=False):
        c_probs = super().forward(x)

        if train and self.interven_prob != 0:
            c_hard = torch.where(c > 0.5, 1.0, 0.0)
            c_probs_mix = self._after_interventions(c_probs, c_true=c_hard)
        else:
            c_probs_mix = c_probs

        y = self.c2y_model(c_probs_mix)
        return c_probs, y

    def _run_step(self, batch, train):
        x, y, c, soft_c, _ = batch
        c_probs, y_logits = self.forward(x, c, train)

        task_loss = self.loss_task(y_logits, y)
        task_loss_scalar = task_loss.detach()

        if self.train_with_c_gt:
            concept_loss = self.loss_concept(c_probs, c)
        else:
            concept_loss = self.loss_concept(c_probs, soft_c)

        loss = self.concept_loss_weight * concept_loss + task_loss
        concept_loss_scalar = concept_loss.detach()

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
