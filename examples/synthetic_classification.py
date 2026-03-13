"""Synthetic hypernode classification with UniGCN and THNN.

Generates a synthetic hypergraph classification task and trains two
models — UniGCNConv and THNNConv — comparing their accuracy.
"""

import equinox as eqx
import hgx
import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_synthetic_data(key, num_nodes=100, num_edges=30, feat_dim=8):
    """Create a synthetic hypernode classification task.

    - 2 classes, ~50 nodes each
    - Class 0 features biased toward [1,0,...], class 1 toward [0,1,...]
    - Mix of intra-class and inter-class hyperedges (sizes 3-8)
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Assign classes: first half = 0, second half = 1
    labels = jnp.concatenate([jnp.zeros(num_nodes // 2),
                              jnp.ones(num_nodes - num_nodes // 2)])

    # Class-correlated features + noise
    base = jnp.zeros((num_nodes, feat_dim))
    base = base.at[:num_nodes // 2, 0].set(1.0)         # class 0 -> dim 0
    base = base.at[num_nodes // 2:, 1].set(1.0)          # class 1 -> dim 1
    noise = 0.3 * jax.random.normal(k1, (num_nodes, feat_dim))
    node_features = base + noise

    # Build hyperedges: ~2/3 intra-class, ~1/3 inter-class
    edges = []
    class0 = jnp.arange(num_nodes // 2)
    class1 = jnp.arange(num_nodes // 2, num_nodes)

    for i in range(num_edges):
        k2, subkey = jax.random.split(k2)
        size = int(jax.random.randint(subkey, (), 3, 9))  # 3 to 8 vertices
        k2, subkey = jax.random.split(k2)

        if i < num_edges * 2 // 3:
            # Intra-class edge: pick from one class
            if i % 2 == 0:
                pool = class0
            else:
                pool = class1
            verts = jax.random.choice(subkey, pool, (size,), replace=False)
        else:
            # Inter-class edge: pick from both classes
            verts = jax.random.choice(subkey, num_nodes, (size,), replace=False)

        edges.append(tuple(int(v) for v in verts))

    hg = hgx.from_edge_list(edges, num_nodes=num_nodes, node_features=node_features)
    return hg, labels.astype(jnp.int32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HyperClassifier(eqx.Module):
    """Two-layer hypergraph classifier: Conv -> ReLU -> Conv -> softmax."""

    conv1: eqx.Module
    conv2: eqx.Module

    def __call__(self, hg):
        # Layer 1
        x = self.conv1(hg)
        x = jax.nn.relu(x)

        # Update hypergraph features for layer 2
        hg = eqx.tree_at(lambda h: h.node_features, hg, x)

        # Layer 2
        x = self.conv2(hg)
        return jax.nn.softmax(x, axis=-1)


def make_model(conv_cls, in_dim, hidden_dim, num_classes, key):
    k1, k2 = jax.random.split(key)
    conv1 = conv_cls(in_dim, hidden_dim, key=k1)
    conv2 = conv_cls(hidden_dim, num_classes, key=k2)
    return HyperClassifier(conv1=conv1, conv2=conv2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def cross_entropy_loss(model, hg, labels):
    probs = model(hg)
    log_probs = jnp.log(probs + 1e-8)
    one_hot = jax.nn.one_hot(labels, num_classes=2)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


@eqx.filter_jit
def train_step(model, hg, labels, opt_state, optimizer):
    loss, grads = eqx.filter_value_and_grad(cross_entropy_loss)(model, hg, labels)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def accuracy(model, hg, labels):
    probs = model(hg)
    preds = jnp.argmax(probs, axis=-1)
    return jnp.mean(preds == labels)


def train(model, hg, labels, num_epochs=50, lr=1e-2):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for epoch in range(num_epochs):
        model, opt_state, loss = train_step(model, hg, labels, opt_state, optimizer)
        if (epoch + 1) % 10 == 0:
            acc = accuracy(model, hg, labels)
            print(f"  Epoch {epoch+1:3d}  loss={loss:.4f}  acc={acc:.2%}")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    key = jax.random.PRNGKey(42)
    k_data, k_uni, k_thnn = jax.random.split(key, 3)

    print("Generating synthetic hypergraph (100 nodes, 30 hyperedges, 2 classes)...")
    hg, labels = make_synthetic_data(k_data)
    print(f"  Nodes: {hg.num_nodes}  Edges: {hg.num_edges}  "
          f"Feature dim: {hg.node_dim}")
    n0 = int(jnp.sum(labels == 0))
    n1 = int(jnp.sum(labels == 1))
    print(f"  Class balance: {n0} / {n1}\n")

    feat_dim = hg.node_dim

    # --- UniGCN ---
    print("=" * 50)
    print("Training UniGCNConv model")
    print("=" * 50)
    model_uni = make_model(hgx.UniGCNConv, feat_dim, 32, 2, k_uni)
    model_uni = train(model_uni, hg, labels)
    final_acc_uni = accuracy(model_uni, hg, labels)
    print(f"  Final accuracy: {final_acc_uni:.2%}\n")

    # --- THNN ---
    print("=" * 50)
    print("Training THNNConv model")
    print("=" * 50)
    model_thnn = make_model(hgx.THNNConv, feat_dim, 32, 2, k_thnn)
    model_thnn = train(model_thnn, hg, labels)
    final_acc_thnn = accuracy(model_thnn, hg, labels)
    print(f"  Final accuracy: {final_acc_thnn:.2%}\n")

    # --- Summary ---
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  UniGCN accuracy:  {final_acc_uni:.2%}")
    print(f"  THNN accuracy:    {final_acc_thnn:.2%}")


if __name__ == "__main__":
    main()
