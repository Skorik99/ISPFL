from federated_methods.base.base import Base

from .base_client import TextBaseClient
from .base_server import TextBaseServer


class TextBase(Base):
    def _init_server(self, cfg):
        self.server = TextBaseServer(cfg)

    def _init_client_cls(self):
        super()._init_client_cls()
        self.client_cls = TextBaseClient
        self.client_kwargs["client_cls"] = self.client_cls
