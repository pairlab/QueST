import torch.nn as nn

class TaskEmbeddingEncoder(nn.Module):
    def __init__(self, n_tasks, embed_dim):
        self.task_encodings = nn.Embedding(n_tasks, embed_dim)

    def forward(self, task_id):
        return self.task_encodings[task_id]