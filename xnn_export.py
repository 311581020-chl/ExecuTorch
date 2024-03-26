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

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())

# quantizer.set_global(quantization_config)
#     .set_object_type(torch.nn.Conv2d, quantization_config) # can configure by module type
#     .set_object_type(torch.nn.functional.linear, quantization_config) # or torch functional op typea
#     .set_module_name("foo.bar", quantization_config)  # or by module fully qualified name

from torch._export import capture_pre_autograd_graph
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge

NUM_CLASSES = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.02
NUM_EPOCH = 10


       
# model = mobilenet_v2()
# model.classifier[1] = nn.Linear(in_features=1280, out_features=10, bias=True)
model = torch.load("../../Lab0/311581020_model.pt",map_location ='cpu')


norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
example_args = (norm(torch.randn(1, 3, 224, 224)),)

# exported_model = capture_pre_autograd_graph(model, example_args)
# prepared_model = prepare_pt2e(exported_model, quantizer)
# print(prepared_model.graph)


edge = to_edge(export(model, example_args))

edge = edge.to_backend(XnnpackPartitioner)

print(edge.exported_program().graph_module)

exec_prog = edge.to_executorch()

with open("xnn_mobilenet.pte", "wb") as file:
    file.write(exec_prog.buffer)
