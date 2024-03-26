import executorch.exir as exir
import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram
from executorch.exir import ExecutorchBackendConfig, ExecutorchProgramManager
from executorch.exir.passes import MemoryPlanningPass

import executorch.exir as exir

NUM_CLASSES = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.02
NUM_EPOCH = 10

# class MyMv2(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.mv2 = mobilenet_v2()
#         self.mv2.classifier[1] = nn.Linear(in_features=1280, out_features=10, bias=True)
#         # self.mv2 =torch.load("Lab0/311581020_model.pt")

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.mv2(x)
       
# model = mobilenet_v2()
# model.classifier[1] = nn.Linear(in_features=1280, out_features=10, bias=True)
model = torch.load("../../Lab0/311581020_model.pt",map_location ='cpu')


# model = model.cuda()
# print(model)
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
example_args = (norm(torch.randn(1, 3, 224, 224)),)

# print(exir.capture(model, example_args).to_edge())
# open("mobilenet.pte", "wb").write(exir.capture(model, example_args).to_edge().to_executorch().buffer)

pre_autograd_aten_dialect = capture_pre_autograd_graph(model, example_args)

print("Pre-Autograd ATen Dialect Graph")
# print(pre_autograd_aten_dialect)

aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, example_args)
print("ATen Dialect Graph")
# print(aten_dialect)

edge_program: exir.EdgeProgramManager = exir.to_edge(aten_dialect)

executorch_program: exir.ExecutorchProgramManager = edge_program.to_executorch(
    ExecutorchBackendConfig(
        passes=[],  # User-defined passes
    )
)
print("ExecuTorch Dialect")
print(executorch_program.exported_program())

with open("mobilenet.pte", "wb") as file:
    file.write(executorch_program.buffer)