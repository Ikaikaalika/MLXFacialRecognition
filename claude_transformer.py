import mlx.core as mx
import mlx.nn as nn
import math

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=112, patch_size=8, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def __call__(self, x):
        # x: (batch_size, in_channels, img_size, img_size)
        x = self.proj(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(0, 2, 1)  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x, mask=None):
        batch_size, seq_length, _ = x.shape
        
        # Linear projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention weights
        # Scaled dot-product attention
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn = attn + mask
            
        attn = mx.softmax(attn, axis=-1)
        attn = self.dropout(attn)
        
        # Apply attention weights to values
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3)
        out = out.reshape(batch_size, seq_length, self.embed_dim)
        
        # Final projection
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class FeedForward(nn.Module):
    """MLP module in transformer block."""
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def __call__(self, x):
        # Layer Normalization + Self-Attention with residual connection
        x = x + self.attn(self.norm1(x))
        # Layer Normalization + MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class FaceViT(nn.Module):
    """Vision Transformer for face recognition."""
    def __init__(
        self,
        img_size=112,
        patch_size=8,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        embedding_dim=512
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Class token and position embeddings
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = nn.Embedding(num_patches + 1, embed_dim)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ]
        
        # Layer normalization and embedding
        self.norm = nn.LayerNorm(embed_dim)
        self.embedding = nn.Linear(embed_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x):
        batch_size = x.shape[0]
        
        # Extract patches and create embeddings
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = mx.broadcast_to(self.cls_token, (batch_size, 1, self.cls_token.shape[2]))
        x = mx.concatenate([cls_tokens, x], axis=1)
        
        # Add position embeddings
        positions = mx.arange(0, x.shape[1])
        pos_embed = self.pos_embed(positions)
        x = x + pos_embed
        
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Get CLS token representation
        x = self.norm(x)
        x = x[:, 0]  # Use CLS token for representation
        
        # Project to embedding dimension
        x = self.embedding(x)
        
        # L2 normalization
        x = x / mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-10)
        
        return x


def train_transformer_model(batch_size=32, epochs=50, learning_rate=0.0001):
    """Complete training function for the Vision Transformer model."""
    from sklearn.datasets import fetch_lfw_people
    import numpy as np
    from sklearn.model_selection import train_test_split
    from PIL import Image
    import random
    from tqdm import tqdm
    import mlx.optimizers as optim

    # Hyperparameters
    MARGIN = 0.2
    IMG_SIZE = 112
    
    # Load LFW dataset
    def load_lfw_dataset(min_faces_per_person=20, resize=True):
        print("Fetching LFW dataset...")
        lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, 
                                     color=True, 
                                     resize=resize if resize else 1.0,
                                     slice_=(slice(0, 250), slice(0, 250)))
        
        print(f"Dataset loaded: {len(lfw_people.images)} images of {len(np.unique(lfw_people.target))} people")
        
        # Organize images by identity
        identities = {}
        for i, (image, target) in enumerate(zip(lfw_people.images, lfw_people.target)):
            person_name = lfw_people.target_names[target]
            if person_name not in identities:
                identities[person_name] = []
            
            # Convert to PIL Image for processing
            img = Image.fromarray(image)
            if resize:
                img = img.resize((IMG_SIZE, IMG_SIZE))
            
            # Store as (image, index) tuple
            identities[person_name].append((img, i))
        
        return identities, lfw_people
    
    # Preprocess image
    def preprocess_image(img):
        # Make sure it's RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if necessary
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE))
        
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        
        # Convert to NCHW format (channels, height, width)
        img_array = np.transpose(img_array, (2, 0, 1))
        return mx.array(img_array).astype(mx.float32)
    
    # Generate triplets
    def generate_triplets(identities, num_triplets=5000):
        triplets = []
        valid_identities = {k: v for k, v in identities.items() if len(v) >= 2}
        identity_list = list(valid_identities.keys())
        
        for _ in range(num_triplets):
            anchor_id = random.choice(identity_list)
            negative_ids = [id for id in identity_list if id != anchor_id]
            negative_id = random.choice(negative_ids)
            
            anchor_positive_imgs = random.sample(valid_identities[anchor_id], 2)
            anchor_img = anchor_positive_imgs[0][0]
            positive_img = anchor_positive_imgs[1][0]
            negative_img = random.choice(valid_identities[negative_id])[0]
            
            triplets.append((anchor_img, positive_img, negative_img))
        
        return triplets
    
    # Create batch triplets
    def create_batch_triplets(triplets, batch_size=32):
        batches = []
        for i in range(0, len(triplets), batch_size):
            batch_triplets = triplets[i:i+batch_size]
            
            anchor_batch = []
            positive_batch = []
            negative_batch = []
            
            for anchor, positive, negative in batch_triplets:
                anchor_batch.append(preprocess_image(anchor))
                positive_batch.append(preprocess_image(positive))
                negative_batch.append(preprocess_image(negative))
            
            anchor_batch = mx.stack(anchor_batch)
            positive_batch = mx.stack(positive_batch)
            negative_batch = mx.stack(negative_batch)
            
            batches.append((anchor_batch, positive_batch, negative_batch))
        
        return batches
    
    # Triplet loss function
    def triplet_loss(anchor, positive, negative, margin=MARGIN):
        pos_dist = mx.sum(mx.square(anchor - positive), axis=-1)
        neg_dist = mx.sum(mx.square(anchor - negative), axis=-1)
        return mx.mean(mx.maximum(pos_dist - neg_dist + margin, 0))
    
    # Main training loop
    # 1. Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # 2. Load dataset
    identities, lfw_people = load_lfw_dataset(min_faces_per_person=20)
    print(f"Dataset contains {len(identities)} people with sufficient images")
    
    # 3. Generate triplets
    all_triplets = generate_triplets(identities, num_triplets=5000)
    
    # 4. Split into train and test sets
    train_triplets, test_triplets = train_test_split(all_triplets, test_size=0.2, random_state=42)
    print(f"Split data into {len(train_triplets)} training and {len(test_triplets)} testing triplets")
    
    # 5. Create batches for training
    train_batches = create_batch_triplets(train_triplets, batch_size=batch_size)
    
    # 6. Initialize transformer model (smaller version for faster training)
    model = FaceViT(
        img_size=IMG_SIZE,
        patch_size=16,  # Larger patches for efficiency
        in_channels=3,
        embed_dim=384,  # Reduced from 768
        depth=6,        # Reduced from 12
        num_heads=6,    # Reduced from 12
        mlp_ratio=4.0,
        dropout=0.1,
        embedding_dim=512
    )
    
    # 7. Set up optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    # 8. Define loss function for batched inputs
    def compute_loss(params, batch):
        anchor_batch, positive_batch, negative_batch = batch
        model.update(params)
        
        anchor_emb = model(anchor_batch)
        pos_emb = model(positive_batch)
        neg_emb = model(negative_batch)
        
        return triplet_loss(anchor_emb, pos_emb, neg_emb)
    
    # 9. Train model
    params = model.parameters()
    
    print("Starting training...")
    for epoch in range(epochs):
        random.shuffle(train_batches)
        epoch_loss = 0
        
        progress_bar = tqdm(train_batches, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            loss, grads = mx.value_and_grad(compute_loss)(params, batch)
            
            updates, optimizer.state = optimizer.update(grads, optimizer.state, params)
            params = optim.apply_updates(params, updates)
            model.update(params)
            
            loss_value = loss.item()
            epoch_loss += loss_value
            progress_bar.set_postfix({"batch_loss": f"{loss_value:.4f}"})
        
        avg_loss = epoch_loss / len(train_batches)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # 10. Evaluate model
    def evaluate_model(model, test_triplets, batch_size=32):
        correct = 0
        total = len(test_triplets)
        
        for i in range(0, total, batch_size):
            batch = test_triplets[i:i+batch_size]
            
            anchor_batch = []
            positive_batch = []
            negative_batch = []
            
            for anchor, positive, negative in batch:
                anchor_batch.append(preprocess_image(anchor))
                positive_batch.append(preprocess_image(positive))
                negative_batch.append(preprocess_image(negative))
            
            anchor_batch = mx.stack(anchor_batch)
            positive_batch = mx.stack(positive_batch)
            negative_batch = mx.stack(negative_batch)
            
            anchor_emb = model(anchor_batch)
            pos_emb = model(positive_batch)
            neg_emb = model(negative_batch)
            
            pos_dist = mx.sum(mx.square(anchor_emb - pos_emb), axis=-1)
            neg_dist = mx.sum(mx.square(anchor_emb - neg_emb), axis=-1)
            
            batch_correct = mx.sum(pos_dist < neg_dist).item()
            correct += batch_correct
        
        accuracy = correct / total
        print(f"Evaluation Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    print("Evaluating model...")
    accuracy = evaluate_model(model, test_triplets)
    
    # 11. Save model
    weights = {k: v.numpy() for k, v in model.parameters().items()}
    np.savez("lfw_transformer_face_recognition.npz", **weights)
    print("Model saved to lfw_transformer_face_recognition.npz")
    
    return model, accuracy

# Uncomment to run training
model, accuracy = train_transformer_model(batch_size=16, epochs=30, learning_rate=0.0001)