"""
Error-Weighted Semantic Coreset Selection (EWSCS) Quantizer.

Novel contribution: Integrates reconstruction error as a first-class signal
in coreset-based codebook re-initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.cluster import MiniBatchKMeans
import numpy as np


class ErrorAwareFeaturePool:
    """
    Feature pool that tracks reconstruction errors for error-guided selection.
    """
    def __init__(self, pool_size, dim=64):
        self.pool_size = pool_size
        self.dim = dim
        self.num_features = 0
        self.features = torch.zeros(pool_size, dim)
        self.errors = torch.zeros(pool_size)
        
    def query(self, features, errors):
        """Add features with their reconstruction errors to the pool."""
        self.features = self.features.to(features.device)
        self.errors = self.errors.to(features.device)
        
        if self.num_features < self.pool_size:
            num_to_add = min(features.size(0), self.pool_size - self.num_features)
            self.features[self.num_features:self.num_features + num_to_add] = features[:num_to_add]
            self.errors[self.num_features:self.num_features + num_to_add] = errors[:num_to_add]
            self.num_features += num_to_add
        else:
            # Reservoir sampling with error-weighted replacement
            num_to_replace = min(features.size(0), self.pool_size)
            random_id = torch.randperm(self.pool_size, device=features.device)[:num_to_replace]
            self.features[random_id] = features[:num_to_replace]
            self.errors[random_id] = errors[:num_to_replace]
            
        return self.features[:self.num_features], self.errors[:self.num_features]


class EWSCSQuantizer(nn.Module):
    """
    Error-Weighted Semantic Coreset Selection Quantizer.
    
    Novel Features:
    1. Reconstruction error as priority signal in codebook updates
    2. Semantic clustering for anatomical feature grouping
    3. Error-weighted stratified sampling across clusters
    4. Adaptive soft re-initialization based on error magnitude
    
    Key Equation:
        score(z_i) = λ * error(z_i) + (1-λ) * min_dist(z_i, E_live)
    
    Where higher scores indicate features that should be prioritized
    for codebook re-initialization.
    """
    def __init__(
        self, 
        num_embeddings: int,
        embedding_dim: int, 
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        # EWSCS-specific parameters
        error_weight: float = 0.7,          # λ in the scoring equation
        error_temperature: float = 0.5,      # Controls adaptive alpha
        soft_reinit_alpha: float = 0.3,      # Blending coefficient
        # Semantic clustering
        num_semantic_clusters: int = 16,
        cluster_update_freq: int = 10,
        # Dead code detection
        dynamic_threshold_factor: float = 0.1,
        # Distance metric
        distance: str = 'l2'
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        # EWSCS parameters
        self.error_weight = error_weight
        self.error_temperature = error_temperature
        self.soft_reinit_alpha = soft_reinit_alpha
        
        # Semantic clustering
        self.num_semantic_clusters = num_semantic_clusters
        self.cluster_update_freq = cluster_update_freq
        self.update_counter = 0
        
        self.dynamic_threshold_factor = dynamic_threshold_factor
        self.distance = distance
        
        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # Usage tracking (EMA)
        self.register_buffer("embed_prob", torch.zeros(num_embeddings))
        
        # Feature pool with error tracking
        self.pool = ErrorAwareFeaturePool(num_embeddings * 4, embedding_dim)
        
        # Clustering
        self.semantic_clusterer = None
        self.cluster_centers = None
        
    def _update_semantic_clusters(self, features):
        """Update semantic clusters using Mini-Batch K-Means."""
        if features.size(0) < self.num_semantic_clusters:
            return
            
        features_np = features.detach().cpu().numpy()
        
        if self.semantic_clusterer is None:
            self.semantic_clusterer = MiniBatchKMeans(
                n_clusters=self.num_semantic_clusters,
                random_state=42,
                batch_size=256,
                max_iter=10
            )
            
        self.semantic_clusterer.partial_fit(features_np)
        
        self.cluster_centers = torch.from_numpy(
            self.semantic_clusterer.cluster_centers_
        ).to(features.device).float()
        
    def _get_cluster_labels(self, features):
        """Fast cluster assignment using cached centers."""
        if self.cluster_centers is None:
            return torch.zeros(features.size(0), dtype=torch.long, device=features.device)
        
        distances = torch.cdist(features, self.cluster_centers)
        return torch.argmin(distances, dim=1)
    
    def _error_weighted_coreset_selection(
        self, 
        features, 
        errors, 
        num_to_select, 
        live_embeddings
    ):
        """
        NOVEL: Error-Weighted Semantic Coreset Selection.
        
        Unlike standard k-means++ which uses only distance, we combine:
        - Reconstruction error (features the model struggles with)
        - Distance to live codebook vectors (diversity)
        
        This creates a self-correcting mechanism where problematic regions
        get priority in codebook updates.
        """
        if len(features) == 0 or num_to_select == 0:
            return torch.empty(0, features.shape[1], device=features.device)
        
        # Normalize errors to [0, 1]
        errors_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
        
        # Compute distance to live codebook (diversity score)
        distances = torch.cdist(features, live_embeddings)
        min_distances = torch.min(distances, dim=1)[0]
        dist_norm = (min_distances - min_distances.min()) / (min_distances.max() - min_distances.min() + 1e-8)
        
        # NOVEL: Combined error-weighted score
        # Higher score = higher priority for selection
        scores = self.error_weight * errors_norm + (1 - self.error_weight) * dist_norm
        
        # Get cluster labels for stratified selection
        cluster_labels = self._get_cluster_labels(features)
        unique_clusters = torch.unique(cluster_labels)
        
        # Compute cluster importance (average error per cluster)
        cluster_importance = torch.zeros(self.num_semantic_clusters, device=features.device)
        cluster_counts = torch.zeros(self.num_semantic_clusters, device=features.device)
        
        for c in unique_clusters:
            mask = (cluster_labels == c)
            cluster_importance[c] = errors_norm[mask].mean()
            cluster_counts[c] = mask.sum()
        
        # Allocate samples per cluster based on importance
        importance_probs = F.softmax(cluster_importance[unique_clusters], dim=0)
        samples_per_cluster = (importance_probs * num_to_select).int()
        
        # Handle rounding
        remainder = num_to_select - samples_per_cluster.sum().item()
        if remainder > 0:
            top_clusters = torch.argsort(cluster_importance[unique_clusters], descending=True)[:remainder]
            samples_per_cluster[top_clusters] += 1
        
        # Select top-scoring features from each cluster
        selected_features = []
        for i, cluster_id in enumerate(unique_clusters):
            n_samples = samples_per_cluster[i].item()
            if n_samples == 0:
                continue
            
            mask = (cluster_labels == cluster_id)
            cluster_features = features[mask]
            cluster_scores = scores[mask]
            
            # Top-k selection within cluster
            k = min(n_samples, len(cluster_scores))
            top_indices = torch.topk(cluster_scores, k)[1]
            selected_features.append(cluster_features[top_indices])
        
        if len(selected_features) == 0:
            # Fallback: global top-k
            top_indices = torch.topk(scores, min(num_to_select, len(scores)))[1]
            return features[top_indices]
        
        result = torch.cat(selected_features, dim=0)
        
        # Ensure exact count
        if len(result) < num_to_select:
            # Pad with more top-scoring features
            remaining_mask = torch.ones(len(features), dtype=torch.bool, device=features.device)
            for sf in selected_features:
                for feat in sf:
                    dists = torch.sum((features - feat.unsqueeze(0)) ** 2, dim=1)
                    closest = torch.argmin(dists)
                    remaining_mask[closest] = False
            
            remaining_scores = scores.clone()
            remaining_scores[~remaining_mask] = -float('inf')
            
            needed = num_to_select - len(result)
            if (remaining_mask).sum() > 0:
                add_indices = torch.topk(remaining_scores, min(needed, remaining_mask.sum()))[1]
                result = torch.cat([result, features[add_indices]], dim=0)
        
        return result[:num_to_select]
    
    def compute_orthogonal_loss(self):
        """Encourage codebook diversity via orthogonal regularization."""
        w = F.normalize(self.embedding.weight, dim=1)
        gram = torch.matmul(w, w.t())
        identity = torch.eye(self.num_embeddings, device=w.device)
        return torch.sum((gram - identity) ** 2) / (self.num_embeddings ** 2)
    
    def forward(self, z):
        """
        Forward pass with EWSCS codebook management.
        
        Returns:
            quantized: Quantized tensor
            loss: VQ loss
            info: (perplexity, encodings, indices, num_alive, avg_error)
        """
        # Reshape: (B, C, H, W) -> (B, H, W, C) -> (N, C)
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Distance calculation
        if self.distance == 'l2':
            distances = (
                torch.sum(z_flattened**2, dim=1, keepdim=True) +
                torch.sum(self.embedding.weight**2, dim=1) -
                2 * torch.matmul(z_flattened, self.embedding.weight.t())
            )
        elif self.distance == 'cos':
            norm_z = F.normalize(z_flattened, dim=1)
            norm_emb = F.normalize(self.embedding.weight, dim=1)
            distances = 1 - torch.matmul(norm_z, norm_emb.t())
        
        # Find nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        quantized = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        
        # Compute per-feature reconstruction error (KEY for EWSCS)
        reconstruction_errors = torch.sum((quantized - z) ** 2, dim=-1).view(-1)
        avg_error = reconstruction_errors.mean()
        
        # VQ losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        quantized = rearrange(quantized, 'b h w c -> b c h w').contiguous()
        
        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        num_alive = self.num_embeddings
        
        if self.training:
            # Update feature pool with errors
            self.pool.query(z_flattened.detach(), reconstruction_errors.detach())
            
            # Update semantic clusters periodically
            self.update_counter += 1
            if self.update_counter % self.cluster_update_freq == 0:
                pool_features, _ = self.pool.query(z_flattened.detach(), reconstruction_errors.detach())
                if pool_features.size(0) >= self.num_semantic_clusters:
                    self._update_semantic_clusters(pool_features)
            
            # EMA usage tracking
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
            
            # Dynamic dead code threshold
            dynamic_threshold = torch.mean(self.embed_prob) * self.dynamic_threshold_factor
            dead_indices = torch.where(self.embed_prob < dynamic_threshold)[0]
            num_dead = len(dead_indices)
            num_alive = self.num_embeddings - num_dead
            
            if num_dead > 0:
                # Get live embeddings
                live_indices = torch.where(self.embed_prob >= dynamic_threshold)[0]
                live_embeddings = self.embedding.weight.data[live_indices]
                
                # Get pool features and errors
                pool_features, pool_errors = self.pool.query(
                    z_flattened.detach(), reconstruction_errors.detach()
                )
                
                # NOVEL: Error-Weighted Semantic Coreset Selection
                new_vectors = self._error_weighted_coreset_selection(
                    pool_features, pool_errors, num_dead, live_embeddings
                )
                
                # Adaptive soft re-initialization (error-scaled)
                adaptive_alpha = self.soft_reinit_alpha * (
                    1 - torch.exp(-avg_error / self.error_temperature)
                )
                
                # Safety: ensure correct size
                if len(new_vectors) < len(dead_indices):
                    needed = len(dead_indices) - len(new_vectors)
                    random_fill = torch.empty(needed, self.embedding_dim, device=new_vectors.device)
                    random_fill.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
                    new_vectors = torch.cat([new_vectors, random_fill], dim=0)
                elif len(new_vectors) > len(dead_indices):
                    new_vectors = new_vectors[:len(dead_indices)]
                
                # Soft re-initialization
                with torch.no_grad():
                    E_dead = self.embedding.weight.data[dead_indices]
                    self.embedding.weight.data[dead_indices] = (
                        E_dead * (1 - adaptive_alpha) + new_vectors * adaptive_alpha
                    )
                    self.embed_prob[dead_indices] = dynamic_threshold
        
        return quantized, loss, (perplexity, encodings, encoding_indices, num_alive, avg_error)
