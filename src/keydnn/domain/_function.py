from abc import ABC, abstractmethod


class Function(ABC):
    @staticmethod
    @abstractmethod
    def forward(ctx, *inputs): ...

    @staticmethod
    @abstractmethod
    def backward(ctx, grad_out): ...
