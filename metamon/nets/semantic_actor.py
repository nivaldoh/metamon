"""
Semantic Actor Head for Pokémon RL that treats actions as entities with descriptive features.

This actor head replaces index-based action selection with a semantic approach that:
1. Extracts rich feature descriptors for each action candidate
2. Encodes descriptors into action embeddings
3. Scores actions via cross-attention with state representations
4. Uses a hierarchical gate for attack/switch decisions
"""

from typing import Optional, Type, Dict, Tuple, List
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from einops import rearrange, repeat
import gin
import numpy as np

from amago.nets.actor_critic import BaseActorHead
from amago.nets.policy_dists import PolicyOutput, Discrete
from amago.nets.ff import MLP, Normalization


# Type effectiveness chart for Pokémon (simplified, gen 1-4 focused)
TYPE_EFFECTIVENESS = {
    "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "steel": 2},
    "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
    "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
    "fighting": {"normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2, "ghost": 0, "dark": 2, "steel": 2, "fairy": 0.5},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0, "fairy": 2},
    "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2, "steel": 2},
    "flying": {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "dark": 0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5, "dark": 2, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2, "steel": 0.5},
    "ghost": {"normal": 0, "psychic": 2, "ghost": 2, "dark": 0.5},
    "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0},
    "dark": {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2, "steel": 0.5, "fairy": 2},
    "fairy": {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "dark": 2, "steel": 0.5},
}


def get_type_effectiveness(attacker_type: str, defender_types: List[str]) -> float:
    """Calculate type effectiveness multiplier for an attack.

    Args:
        attacker_type: The type of the attacking move
        defender_types: List of defender's types (1 or 2 types)

    Returns:
        Effectiveness multiplier (0, 0.25, 0.5, 1, 2, 4)
    """
    if attacker_type not in TYPE_EFFECTIVENESS:
        return 1.0

    multiplier = 1.0
    type_chart = TYPE_EFFECTIVENESS[attacker_type]

    for def_type in defender_types:
        if def_type in type_chart:
            multiplier *= type_chart[def_type]
        # If not in chart, neutral effectiveness (1.0)

    return multiplier


def encode_type_as_onehot(type_name: str, type_list: List[str]) -> torch.Tensor:
    """Encode a type name as a one-hot vector.

    Args:
        type_name: Name of the type
        type_list: List of all possible types

    Returns:
        One-hot encoded tensor
    """
    if type_name in type_list:
        idx = type_list.index(type_name)
        onehot = torch.zeros(len(type_list))
        onehot[idx] = 1.0
        return onehot
    else:
        # Unknown type - return zero vector
        return torch.zeros(len(type_list))


def estimate_damage_range(base_power: int, attacker_level: int = 50,
                         effectiveness: float = 1.0, stab: bool = False) -> Tuple[float, float]:
    """Estimate damage range as percentage of opponent's HP.

    Simplified damage calculation for feature extraction.

    Args:
        base_power: Base power of the move
        attacker_level: Level of the attacker (default 50)
        effectiveness: Type effectiveness multiplier
        stab: Whether move gets STAB bonus

    Returns:
        (min_damage_pct, max_damage_pct) as fractions of opponent HP
    """
    if base_power <= 0:
        return (0.0, 0.0)

    # Simplified damage formula (assumes average stats)
    base_damage = ((2 * attacker_level + 10) / 250) * base_power * 1.0  # Assuming stat ratio ~1

    if stab:
        base_damage *= 1.5

    base_damage *= effectiveness

    # Random multiplier range in Pokémon is 0.85-1.0
    min_damage = base_damage * 0.85
    max_damage = base_damage * 1.0

    # Convert to percentage (rough estimate)
    # Assuming average HP is around 200-300 at level 50
    avg_hp = 250
    min_pct = min(min_damage / avg_hp, 1.0)
    max_pct = min(max_damage / avg_hp, 1.0)

    return (min_pct, max_pct)


def safe_softmax(logits: torch.Tensor, dim: int = -1, eps: float = 1e-10) -> torch.Tensor:
    """Numerically stable softmax that handles all-inf inputs.

    When all values along a dimension are -inf (e.g., all actions illegal),
    returns zeros instead of NaN.

    Args:
        logits: Input logits tensor
        dim: Dimension along which to apply softmax
        eps: Small epsilon for numerical stability

    Returns:
        Probabilities that sum to 1 (or 0 if all inputs were -inf)
    """
    # Check if all values in the dimension are -inf
    is_all_masked = (logits == float('-inf')).all(dim=dim, keepdim=True)

    # For numerical stability, subtract max (but handle all-inf case)
    logits_max = logits.max(dim=dim, keepdim=True).values
    # Where all are -inf, set max to 0 to avoid inf - inf = nan
    logits_max = torch.where(is_all_masked, torch.zeros_like(logits_max), logits_max)

    # Compute exp(logits - max)
    logits_exp = torch.exp(logits - logits_max)

    # Sum and normalize
    logits_sum = logits_exp.sum(dim=dim, keepdim=True) + eps
    probs = logits_exp / logits_sum

    # Where all inputs were -inf, set output to zeros
    probs = torch.where(is_all_masked, torch.zeros_like(probs), probs)

    return probs


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross attention for scoring actions based on state.

    The state representation serves as the query, while action embeddings
    serve as keys. We only compute attention scores, not value-weighted outputs,
    since we need scores per action rather than aggregated features.
    """

    def __init__(
        self,
        state_dim: int,
        action_emb_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert action_emb_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = action_emb_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Project state to query
        self.q_proj = nn.Linear(state_dim, action_emb_dim)
        # Project action embeddings to keys (no values needed)
        self.k_proj = nn.Linear(action_emb_dim, action_emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        state: torch.Tensor,  # [batch, seq_len, state_dim]
        action_embs: torch.Tensor,  # [batch, seq_len, n_actions, emb_dim]
    ) -> torch.Tensor:  # [batch, seq_len, n_actions]
        B, L, _ = state.shape
        _, _, N, _ = action_embs.shape

        # Compute queries from state
        Q = self.q_proj(state)  # [B, L, emb_dim]
        Q = rearrange(Q, "b l (h d) -> b h l d", h=self.num_heads)

        # Compute keys from action embeddings
        K = self.k_proj(action_embs)  # [B, L, N, emb_dim]
        K = rearrange(K, "b l n (h d) -> b h l n d", h=self.num_heads)

        # Attention scores (no value computation needed)
        scores = torch.einsum("bhld,bhlnd->bhln", Q, K) / self.scale

        # Average scores over heads
        scores = scores.mean(dim=1)  # [B, L, N]

        return scores


@gin.configurable
class SemanticActorHead(BaseActorHead):
    """Semantic actor head that scores actions based on learned feature representations.

    This actor treats each action as an entity with descriptive features rather than
    just an index. It extracts features for moves (type, power, accuracy, etc.) and
    switches (target stats, type matchups, etc.), encodes them into embeddings, and
    uses cross-attention to score actions based on the current state.

    Args:
        state_dim: Dimension of state representations from the trajectory encoder
        action_dim: Number of actions (should be 9 for Pokémon: 4 moves + 5 switches)
        discrete: Must be True for Pokémon action space
        gammas: Multi-gamma discount factors for AMAGO training

    Keyword Args:
        descriptor_hidden_dim: Hidden dimension for descriptor encoder
        action_emb_dim: Dimension of encoded action embeddings
        num_attention_heads: Number of heads for cross-attention scoring
        use_gate: Whether to use hierarchical attack/switch gate
        use_bilinear_scoring: Use bilinear scoring instead of cross-attention
        dropout: Dropout rate for attention and networks
        normalize_descriptors: Whether to normalize descriptor features
        cache_embeddings: Cache action embeddings when they don't change
        fallback_mlp: Include fallback MLP for baseline comparison
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        descriptor_hidden_dim: int = 128,
        action_emb_dim: int = 64,
        num_attention_heads: int = 4,
        use_gate: bool = True,
        use_bilinear_scoring: bool = False,
        dropout: float = 0.1,
        normalize_descriptors: bool = True,
        cache_embeddings: bool = True,
        fallback_mlp: bool = False,
    ):
        assert discrete, "SemanticActorHead only supports discrete actions"
        assert action_dim == 9, "Expected 9 actions (4 moves + 5 switches)"

        # Use Discrete distribution for compatibility
        continuous_dist_type = None  # Not used for discrete actions

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=gammas,
            continuous_dist_type=Discrete,
        )

        self.descriptor_hidden_dim = descriptor_hidden_dim
        self.action_emb_dim = action_emb_dim
        self.use_gate = use_gate
        self.use_bilinear_scoring = use_bilinear_scoring
        self.normalize_descriptors = normalize_descriptors
        self.cache_embeddings = cache_embeddings
        self.fallback_mlp = fallback_mlp

        # Descriptor dimensions (will be determined dynamically)
        self.move_descriptor_dim = None
        self.switch_descriptor_dim = None

        # Descriptor encoders - initialized lazily after we know descriptor dims
        self.move_encoder = None
        self.switch_encoder = None

        # Scoring mechanism
        if use_bilinear_scoring:
            # Will be initialized after we know dimensions
            self.bilinear_W = None
        else:
            self.cross_attention = MultiHeadCrossAttention(
                state_dim=state_dim,
                action_emb_dim=action_emb_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
            )

        # Attack/switch gate
        if use_gate:
            self.gate_head = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2),  # [attack_logit, switch_logit]
            )

        # Optional fallback MLP for baseline comparison
        if fallback_mlp:
            self.fallback_actor = MLP(
                d_inp=state_dim,
                d_hidden=256,
                n_layers=2,
                d_output=action_dim * len(gammas),
                dropout_p=dropout,
                activation="leaky_relu",
            )

        # Cache for action embeddings
        self._embedding_cache = {}
        self._cache_key = None

    def _init_descriptor_encoders(self, move_dim: int, switch_dim: int):
        """Initialize descriptor encoders once we know the dimensions."""
        if self.move_encoder is not None:
            return  # Already initialized

        self.move_descriptor_dim = move_dim
        self.switch_descriptor_dim = switch_dim

        # Move descriptor encoder
        self.move_encoder = nn.Sequential(
            nn.Linear(move_dim, self.descriptor_hidden_dim),
            nn.LayerNorm(self.descriptor_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.descriptor_hidden_dim, self.action_emb_dim),
            nn.LayerNorm(self.action_emb_dim),
        )

        # Switch descriptor encoder (may have different input dim)
        self.switch_encoder = nn.Sequential(
            nn.Linear(switch_dim, self.descriptor_hidden_dim),
            nn.LayerNorm(self.descriptor_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.descriptor_hidden_dim, self.action_emb_dim),
            nn.LayerNorm(self.action_emb_dim),
        )

        # Initialize bilinear matrix if needed
        if self.use_bilinear_scoring:
            # Proper shape: [state_dim, action_emb_dim, 1] for output dimension
            self.bilinear_W = nn.Parameter(
                torch.randn(self.state_dim, self.action_emb_dim, 1) * 0.01
            )

    def extract_action_descriptors(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract feature descriptors for moves and switches from observations.

        This method attempts to extract semantic features from the observation dictionary.
        It looks for pre-computed features first, then falls back to extracting from
        text/numerical observations if available, and finally uses placeholder features.

        Args:
            obs: Dictionary of observations containing battle state information

        Returns:
            move_descriptors: [batch, seq_len, 4, descriptor_dim]
            switch_descriptors: [batch, seq_len, 5, descriptor_dim]
        """
        # Get batch dimensions
        if 'action_mask' in obs:
            B, L = obs['action_mask'].shape[:2]
        elif 'illegal_actions' in obs:
            B, L = obs['illegal_actions'].shape[:2]
        else:
            # Try to infer from any key
            for key, val in obs.items():
                if isinstance(val, torch.Tensor) and val.dim() >= 2:
                    B, L = val.shape[:2]
                    break
            else:
                raise ValueError("Could not infer batch dimensions from observations")

        device = next(iter(obs.values())).device

        # Extract move descriptors
        move_descriptors = self._extract_move_descriptors(obs, B, L, device)

        # Extract switch descriptors
        switch_descriptors = self._extract_switch_descriptors(obs, B, L, device)

        # Normalize if requested
        if self.normalize_descriptors:
            move_descriptors = F.normalize(move_descriptors, dim=-1)
            switch_descriptors = F.normalize(switch_descriptors, dim=-1)

        return move_descriptors, switch_descriptors

    def _extract_move_descriptors(
        self, obs: Dict[str, torch.Tensor], B: int, L: int, device: torch.device
    ) -> torch.Tensor:
        """Extract move feature descriptors from observations.

        Features extracted (when available):
        - Move type (one-hot encoded)
        - Base power (normalized)
        - Accuracy
        - Priority (normalized)
        - Category (physical/special/status as one-hot)
        - PP remaining (normalized)
        - STAB indicator
        - Type effectiveness
        - Damage estimates

        Args:
            obs: Observation dictionary
            B: Batch size
            L: Sequence length
            device: Torch device

        Returns:
            move_descriptors: [B, L, 4, feature_dim]
        """
        move_features = []

        # List of all Pokémon types for encoding
        type_list = list(TYPE_EFFECTIVENESS.keys())
        n_types = len(type_list)

        # Try to extract pre-computed move features
        precomputed_keys = [
            'move_type',      # Type of each move (one-hot or embedding)
            'move_power',     # Base power
            'move_accuracy',  # Accuracy
            'move_priority',  # Priority
            'move_category',  # Physical/Special/Status
            'move_pp',        # PP remaining
            'move_stab',      # STAB indicator
            'move_effectiveness',  # Type effectiveness vs opponent
        ]

        has_precomputed = any(k in obs for k in precomputed_keys[:3])

        if has_precomputed:
            # Use pre-computed features
            for key in precomputed_keys:
                if key in obs:
                    feat = obs[key]  # [B, L, 4] or [B, L, 4, dim]
                    if feat.dim() == 3:
                        feat = feat.unsqueeze(-1)
                    move_features.append(feat)
        else:
            # Generate placeholder features with reasonable dimensions
            # Type encoding (18 types as one-hot)
            type_features = torch.zeros(B, L, 4, n_types, device=device)
            move_features.append(type_features)

            # Base power (normalized to 0-1)
            power_features = torch.rand(B, L, 4, 1, device=device)
            move_features.append(power_features)

            # Accuracy (0-1)
            accuracy_features = torch.ones(B, L, 4, 1, device=device) * 0.9
            move_features.append(accuracy_features)

            # Priority (-7 to +5, normalized)
            priority_features = torch.zeros(B, L, 4, 1, device=device)
            move_features.append(priority_features)

            # Category (3-dim one-hot: physical, special, status)
            category_features = torch.zeros(B, L, 4, 3, device=device)
            category_features[..., 0] = 1.0  # Default to physical
            move_features.append(category_features)

            # PP (normalized 0-1)
            pp_features = torch.ones(B, L, 4, 1, device=device)
            move_features.append(pp_features)

            # STAB indicator
            stab_features = torch.zeros(B, L, 4, 1, device=device)
            move_features.append(stab_features)

            # Type effectiveness (log scale)
            effectiveness_features = torch.zeros(B, L, 4, 1, device=device)
            move_features.append(effectiveness_features)

            # Damage estimates (min, max as percentages)
            damage_features = torch.zeros(B, L, 4, 2, device=device)
            move_features.append(damage_features)

        if move_features:
            move_descriptors = torch.cat(move_features, dim=-1)
        else:
            # Final fallback: random features
            feature_dim = n_types + 10  # Type encoding + other features
            move_descriptors = torch.randn(B, L, 4, feature_dim, device=device) * 0.1

        return move_descriptors

    def _extract_switch_descriptors(
        self, obs: Dict[str, torch.Tensor], B: int, L: int, device: torch.device
    ) -> torch.Tensor:
        """Extract switch target feature descriptors from observations.

        Features extracted (when available):
        - Target HP percentage
        - Target types (one-hot encoded)
        - Status condition
        - Stat stages
        - Type advantage vs opponent
        - Speed tier
        - Has priority moves

        Args:
            obs: Observation dictionary
            B: Batch size
            L: Sequence length
            device: Torch device

        Returns:
            switch_descriptors: [B, L, 5, feature_dim]
        """
        switch_features = []

        # List of all Pokémon types for encoding
        type_list = list(TYPE_EFFECTIVENESS.keys())
        n_types = len(type_list)

        # Try to extract pre-computed switch features
        precomputed_keys = [
            'switch_hp',          # HP percentage
            'switch_types',       # Type(s) of switch target
            'switch_status',      # Status condition
            'switch_atk_stage',   # Attack stage
            'switch_def_stage',   # Defense stage
            'switch_spa_stage',   # Special Attack stage
            'switch_spd_stage',   # Special Defense stage
            'switch_spe_stage',   # Speed stage
            'switch_type_advantage',  # Type advantage vs opponent
            'switch_speed_tier',  # Relative speed
            'switch_has_priority',  # Has priority moves
        ]

        has_precomputed = any(k in obs for k in precomputed_keys[:3])

        if has_precomputed:
            # Use pre-computed features
            for key in precomputed_keys:
                if key in obs:
                    feat = obs[key]  # [B, L, 5] or [B, L, 5, dim]
                    if feat.dim() == 3:
                        feat = feat.unsqueeze(-1)
                    switch_features.append(feat)
        else:
            # Generate placeholder features with reasonable dimensions
            # HP percentage
            hp_features = torch.rand(B, L, 5, 1, device=device)
            switch_features.append(hp_features)

            # Type encoding (2 * n_types for dual types)
            type_features = torch.zeros(B, L, 5, 2 * n_types, device=device)
            switch_features.append(type_features)

            # Status (6-dim: none, burn, freeze, paralysis, poison, sleep)
            status_features = torch.zeros(B, L, 5, 6, device=device)
            status_features[..., 0] = 1.0  # Default to no status
            switch_features.append(status_features)

            # Stat stages (6 stats, normalized -6 to +6)
            stat_features = torch.zeros(B, L, 5, 6, device=device)
            switch_features.append(stat_features)

            # Type advantage (defensive and offensive)
            type_adv_features = torch.zeros(B, L, 5, 2, device=device)
            switch_features.append(type_adv_features)

            # Speed tier (-1 to 1, where negative means slower)
            speed_features = torch.zeros(B, L, 5, 1, device=device)
            switch_features.append(speed_features)

            # Has priority moves indicator
            priority_features = torch.zeros(B, L, 5, 1, device=device)
            switch_features.append(priority_features)

            # Role score (offensive vs defensive)
            role_features = torch.zeros(B, L, 5, 2, device=device)
            switch_features.append(role_features)

        if switch_features:
            switch_descriptors = torch.cat(switch_features, dim=-1)
        else:
            # Final fallback: random features
            feature_dim = 2 * n_types + 20  # Types + other features
            switch_descriptors = torch.randn(B, L, 5, feature_dim, device=device) * 0.1

        return switch_descriptors

    def encode_action_descriptors(
        self,
        move_descriptors: torch.Tensor,
        switch_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        """Encode raw descriptors into action embeddings.

        Args:
            move_descriptors: [batch, seq_len, 4, move_descriptor_dim]
            switch_descriptors: [batch, seq_len, 5, switch_descriptor_dim]

        Returns:
            action_embeddings: [batch, seq_len, 9, action_emb_dim]
        """
        B, L, _, move_dim = move_descriptors.shape
        _, _, _, switch_dim = switch_descriptors.shape

        # Initialize encoders if needed
        if self.move_encoder is None:
            self._init_descriptor_encoders(move_dim, switch_dim)

        # Reshape for encoding
        move_desc_flat = rearrange(move_descriptors, "b l n d -> (b l n) d")
        switch_desc_flat = rearrange(switch_descriptors, "b l n d -> (b l n) d")

        # Encode
        move_embs = self.move_encoder(move_desc_flat)  # [(B*L*4), emb_dim]
        switch_embs = self.switch_encoder(switch_desc_flat)  # [(B*L*5), emb_dim]

        # Reshape back
        move_embs = rearrange(move_embs, "(b l n) d -> b l n d", b=B, l=L, n=4)
        switch_embs = rearrange(switch_embs, "(b l n) d -> b l n d", b=B, l=L, n=5)

        # Concatenate all action embeddings
        action_embs = torch.cat([move_embs, switch_embs], dim=2)  # [B, L, 9, emb_dim]

        return action_embs

    def score_actions(
        self,
        state: torch.Tensor,
        action_embs: torch.Tensor,
    ) -> torch.Tensor:
        """Score each action based on state representation.

        Args:
            state: [batch, seq_len, state_dim]
            action_embs: [batch, seq_len, 9, action_emb_dim]

        Returns:
            scores: [batch, seq_len, 9]
        """
        if self.use_bilinear_scoring:
            # Bilinear scoring: s^T W a
            # state: [B, L, state_dim]
            # action_embs: [B, L, 9, emb_dim]
            # W: [state_dim, emb_dim, 1]
            # Use unique dimension labels to avoid collision
            scores = torch.einsum(
                "bls,bne,seo->bln",
                state,
                action_embs,
                self.bilinear_W
            ).squeeze(-1)  # Remove the output dimension
        else:
            # Cross-attention scoring
            scores = self.cross_attention(state, action_embs)

        return scores

    def actor_network_forward(
        self,
        state: torch.Tensor,
        log_dict: Optional[dict] = None,
        straight_from_obs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass of the semantic actor.

        Args:
            state: State representation from trajectory encoder [B, L, state_dim]
            log_dict: Optional dict for logging
            straight_from_obs: Raw observations for descriptor extraction

        Returns:
            dist_params: [B, L, n_gammas, 9] logits for action distribution
        """
        B, L, D = state.shape
        device = state.device

        # Fallback to MLP if requested and no observations provided
        if self.fallback_mlp and straight_from_obs is None:
            logits = self.fallback_actor(state)
            logits = rearrange(logits, "b l (g a) -> b l g a", g=self.num_gammas)
            return logits

        # Extract observations
        if straight_from_obs is None:
            # Create dummy observations for compatibility
            straight_from_obs = {
                'action_mask': torch.ones(B, L, 9, device=device),
            }

        # Extract action descriptors
        move_desc, switch_desc = self.extract_action_descriptors(straight_from_obs)

        # Check if we can use cached embeddings
        cache_key = (move_desc.shape, switch_desc.shape, device)
        if self.cache_embeddings and cache_key == self._cache_key:
            # Check if descriptors haven't changed (approximate check)
            if 'move_desc' in self._embedding_cache:
                old_move = self._embedding_cache['move_desc']
                old_switch = self._embedding_cache['switch_desc']
                if torch.allclose(old_move, move_desc, atol=1e-6) and \
                   torch.allclose(old_switch, switch_desc, atol=1e-6):
                    action_embs = self._embedding_cache['embeddings']
                else:
                    # Descriptors changed, recompute
                    action_embs = self.encode_action_descriptors(move_desc, switch_desc)
                    self._embedding_cache = {
                        'move_desc': move_desc.detach(),
                        'switch_desc': switch_desc.detach(),
                        'embeddings': action_embs.detach(),
                    }
            else:
                # First time, compute and cache
                action_embs = self.encode_action_descriptors(move_desc, switch_desc)
                self._embedding_cache = {
                    'move_desc': move_desc.detach(),
                    'switch_desc': switch_desc.detach(),
                    'embeddings': action_embs.detach(),
                }
        else:
            # No caching or cache miss
            action_embs = self.encode_action_descriptors(move_desc, switch_desc)
            if self.cache_embeddings:
                self._cache_key = cache_key
                self._embedding_cache = {
                    'move_desc': move_desc.detach(),
                    'switch_desc': switch_desc.detach(),
                    'embeddings': action_embs.detach(),
                }

        # Score actions
        scores = self.score_actions(state, action_embs)  # [B, L, 9]

        # Apply legality mask
        if 'action_mask' in straight_from_obs:
            mask = straight_from_obs['action_mask']  # [B, L, 9]
            legality_mask = torch.where(mask == 1, 0.0, -float('inf'))
            scores = scores + legality_mask

        # Optionally use hierarchical gate
        if self.use_gate:
            gate_logits = self.gate_head(state)  # [B, L, 2]
            gate_probs = F.softmax(gate_logits, dim=-1)

            # Split scores into moves and switches
            move_scores = scores[..., :4]  # [B, L, 4]
            switch_scores = scores[..., 4:]  # [B, L, 5]

            # Apply safe softmax to handle all-illegal cases
            move_probs = safe_softmax(move_scores, dim=-1)
            switch_probs = safe_softmax(switch_scores, dim=-1)

            # Compose final distribution
            final_probs = torch.cat([
                gate_probs[..., 0:1] * move_probs,  # P(attack) * P(move|attack)
                gate_probs[..., 1:2] * switch_probs,  # P(switch) * P(target|switch)
            ], dim=-1)

            # Convert back to logits for compatibility with Discrete distribution
            # Add small epsilon to prevent log(0)
            # Handle case where final_probs could be all zeros
            eps = 1e-10
            logits = torch.log(final_probs + eps)

            # If all actions were illegal, the logits will be very negative but not -inf
            # This is fine as the Discrete distribution will handle it
        else:
            # Direct softmax over all actions
            logits = scores

        # Expand for multi-gamma (same policy for all gammas)
        logits = repeat(logits, "b l a -> b l g a", g=self.num_gammas)

        if log_dict is not None:
            # Log attention weights or other metrics
            with torch.no_grad():
                valid_moves = (scores[..., :4] > -float('inf')).any(dim=-1).float().mean()
                valid_switches = (scores[..., 4:] > -float('inf')).any(dim=-1).float().mean()

                log_dict['semantic_actor/mean_move_score'] = scores[..., :4][scores[..., :4] > -float('inf')].mean().item() if valid_moves > 0 else 0.0
                log_dict['semantic_actor/mean_switch_score'] = scores[..., 4:][scores[..., 4:] > -float('inf')].mean().item() if valid_switches > 0 else 0.0
                log_dict['semantic_actor/valid_move_positions'] = valid_moves.item()
                log_dict['semantic_actor/valid_switch_positions'] = valid_switches.item()

                if self.use_gate:
                    log_dict['semantic_actor/mean_attack_prob'] = gate_probs[..., 0].mean().item()
                    log_dict['semantic_actor/mean_switch_prob'] = gate_probs[..., 1].mean().item()

        return logits