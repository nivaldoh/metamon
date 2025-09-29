# Semantic Actor Head for Pokémon RL

## Overview

The `SemanticActorHead` is an advanced actor architecture that treats actions as entities with rich feature descriptors rather than abstract indices. This approach enables better generalization to new Pokémon and moves by reasoning about action properties semantically.

## Key Features

### 1. Action Descriptor Extraction
Extracts semantic features for each action candidate:

**Move Descriptors:**
- Type (one-hot encoded, 18 dimensions)
- Base power (normalized 0-1)
- Accuracy (0-1)
- Priority (-7 to +5, normalized)
- Category (physical/special/status as 3-dim one-hot)
- PP remaining (normalized)
- STAB indicator
- Type effectiveness against opponent
- Damage estimates (min/max as percentages)

**Switch Descriptors:**
- Target HP percentage
- Target types (dual type encoding)
- Status condition (6-dim: none, burn, freeze, paralysis, poison, sleep)
- Stat stages (6 stats, normalized -6 to +6)
- Type advantage (defensive and offensive)
- Speed tier relative to opponent
- Has priority moves indicator
- Role score (offensive vs defensive)

### 2. Descriptor Encoding
Small neural networks (`g(·)`) encode raw descriptors into action embeddings:
- Separate encoders for moves and switches
- Layer normalization for stability
- Configurable hidden dimensions

### 3. Cross-Attention Scoring
Actions are scored via multi-head cross-attention:
- State representation as query
- Action embeddings as keys
- Produces attention-weighted scores for each action
- Alternative: Bilinear scoring for efficiency

### 4. Hierarchical Gate
Optional binary gate for high-level decision making:
- First decides: attack or switch?
- Then selects specific move or switch target
- Improves learning stability through curriculum

### 5. Caching Mechanism
Efficiency optimization:
- Caches action embeddings when descriptors don't change
- Reduces redundant computation
- Particularly useful during evaluation

## Usage

### Training with Semantic Actor

1. **Imitation Learning:**
```bash
python -m metamon.rl.train \
    --run_name semantic_il_experiment \
    --model_gin_config semantic_small_agent.gin \
    --train_gin_config semantic_il.gin \
    --save_dir ~/ckpts/ \
    --log
```

2. **Reinforcement Learning:**
```bash
python -m metamon.rl.train \
    --run_name semantic_rl_experiment \
    --model_gin_config semantic_agent.gin \
    --train_gin_config exp_rl.gin \
    --save_dir ~/ckpts/ \
    --log
```

### Configuration Files

**Model Configurations:**
- `semantic_small_agent.gin`: Lightweight model for fast training
- `semantic_agent.gin`: Standard model with balanced parameters
- `semantic_medium_agent.gin`: Larger model for better performance

**Training Configurations:**
- `semantic_il.gin`: Imitation learning setup
- Standard RL configs (`exp_rl.gin`, `binary_rl.gin`) work with semantic models

### Key Hyperparameters

Configure in `.gin` files:
```python
# Semantic actor specific
MetamonSemanticActor.descriptor_hidden_dim = 128  # Descriptor encoder hidden size
MetamonSemanticActor.action_emb_dim = 64         # Action embedding dimension
MetamonSemanticActor.num_attention_heads = 4     # Cross-attention heads
MetamonSemanticActor.use_gate = True            # Enable hierarchical gate
MetamonSemanticActor.normalize_descriptors = True # L2 normalize descriptors
MetamonSemanticActor.cache_embeddings = True    # Cache for efficiency
```

## Architecture Details

### Forward Pass Flow
1. **Extract Descriptors**: Parse observations to build feature vectors for each action
2. **Encode**: Transform descriptors into action embeddings via MLPs
3. **Score**: Compute attention between state and action embeddings
4. **Gate** (optional): Apply hierarchical decision gate
5. **Mask**: Apply legality constraints to ensure valid actions
6. **Output**: Return action distribution parameters

### Integration with AMAGO
- Inherits from `amago.nets.actor_critic.BaseActorHead`
- Compatible with multi-gamma training
- Supports illegal action masking via `MetamonSemanticActor` wrapper
- Works with existing replay buffers and training loops

## Expected Benefits

1. **Better Generalization**: Understands actions by their properties, not memorized indices
2. **Interpretability**: Attention weights reveal which action features drive decisions
3. **Transfer Learning**: Knowledge transfers to new Pokémon with similar move properties
4. **Curriculum Learning**: Gate naturally learns high-level strategy before details
5. **Compositional Reasoning**: Can reason about novel type combinations

## Implementation Files

- `/metamon/nets/semantic_actor.py`: Core `SemanticActorHead` implementation
- `/metamon/rl/metamon_to_amago.py`: `MetamonSemanticActor` wrapper with masking
- `/metamon/rl/configs/models/semantic_*.gin`: Model configurations
- `/metamon/rl/configs/training/semantic_*.gin`: Training configurations

## Future Extensions

Potential improvements to explore:
1. **Richer Descriptors**: Add move effects, abilities, items
2. **Learned Embeddings**: Replace one-hot type encoding with learned embeddings
3. **Temporal Features**: Include move history, PP depletion rates
4. **Opponent Modeling**: Encode opponent's revealed moves and patterns
5. **Multi-Scale Attention**: Different attention patterns for tactical vs strategic decisions