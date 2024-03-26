import torch
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram


class SimpleConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv(x)
        return self.relu(a)


example_args = (torch.randn(1, 3, 256, 256),)
pre_autograd_aten_dialect = capture_pre_autograd_graph(SimpleConv(), example_args)
print("Pre-Autograd ATen Dialect Graph")
print(pre_autograd_aten_dialect)

aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, example_args)
print("ATen Dialect Graph")
print(aten_dialect)