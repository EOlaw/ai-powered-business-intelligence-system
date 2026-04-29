"""
InsightSerenity AI Engine — Learning Paradigms Package
=======================================================
Full ML capability spectrum beyond the core LLM.

Supervised:
    from src.learning import NeuralClassifier, ClassifierConfig
    from src.learning import NeuralRegressor, LinearRegressor, PolynomialRegressor
    from src.learning import SequenceLabeler, CRF

Unsupervised:
    from src.learning import KMeans, DBSCAN, AgglomerativeClustering
    from src.learning import Autoencoder, DenoisingAutoencoder
    from src.learning import VAE, VAEConfig
    from src.learning import PCA, TSNE, UMAP

Self-supervised:
    from src.learning import SimCLR, NTXentLoss, CLIPStyleModel
    from src.learning import MaskedAutoencoder, BERTStylePretrainer
"""

# Supervised
from src.learning.supervised.classification import (
    NeuralClassifier, ClassifierConfig,
    binary_classifier, multiclass_classifier, multilabel_classifier,
)
from src.learning.supervised.regression import (
    NeuralRegressor, RegressorConfig,
    LinearRegressor, PolynomialRegressor,
)
from src.learning.supervised.sequence import (
    SequenceLabeler, SequenceLabelerConfig, CRF,
)

# Unsupervised
from src.learning.unsupervised.clustering import (
    KMeans, DBSCAN, AgglomerativeClustering,
)
from src.learning.unsupervised.autoencoder import (
    Autoencoder, DenoisingAutoencoder, AutoencoderConfig,
)
from src.learning.unsupervised.vae import VAE, VAEConfig, VAEEncoder
from src.learning.unsupervised.dimensionality import PCA, TSNE, UMAP

# Self-supervised
from src.learning.self_supervised.contrastive import (
    SimCLR, SimCLRConfig, NTXentLoss, ProjectionHead,
    CLIPStyleModel, feature_augment,
)
from src.learning.self_supervised.masked import (
    MaskedAutoencoder, MAEConfig, BERTStylePretrainer,
)

__all__ = [
    # Supervised
    "NeuralClassifier", "ClassifierConfig",
    "binary_classifier", "multiclass_classifier", "multilabel_classifier",
    "NeuralRegressor", "RegressorConfig",
    "LinearRegressor", "PolynomialRegressor",
    "SequenceLabeler", "SequenceLabelerConfig", "CRF",
    # Unsupervised
    "KMeans", "DBSCAN", "AgglomerativeClustering",
    "Autoencoder", "DenoisingAutoencoder", "AutoencoderConfig",
    "VAE", "VAEConfig", "VAEEncoder",
    "PCA", "TSNE", "UMAP",
    # Self-supervised
    "SimCLR", "SimCLRConfig", "NTXentLoss", "ProjectionHead",
    "CLIPStyleModel", "feature_augment",
    "MaskedAutoencoder", "MAEConfig", "BERTStylePretrainer",
]
