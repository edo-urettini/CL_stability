#!/usr/bin/env python3
from .icarl import OnlineICaRL, OnlineICaRLLossPlugin
from .erace import ER_ACE
from .lwf import LwFPlugin
from .agem import AGEMPlugin
from .ng import NGPlugin, WeightedCrossEntropyLossPlugin
from .init_embedding import InitEmbeddingPlugin
from .robust_grad import SignSGDPlugin
from .ema import MeanEvaluation