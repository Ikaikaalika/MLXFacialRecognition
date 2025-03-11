import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import random
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from PIL import Image
import io
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
MARGIN = 0.2
IMG_SIZE = 112  # Common size for face recognition

class ImprovedFaceCNN(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Initial block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(64, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.res_block3 = ResidualBlock(256, 512)
        
        # Attention mechanism
        self.attention = SpatialAttention(512)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding layers with dropout
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, embedding_dim)
        
        self.embedding_dim = embedding_dim

    def __call__(self, x):
        # Initial convolution
        x = nn.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks with progressive downsampling
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.reshape(x.shape[0], -1)
        
        # Embedding layers
        x = nn.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2 normalization
        x = x / mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-10)
        
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        # Pooling for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def __call__(self, x):
        residual = self.shortcut(x)
        
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        x += residual
        x = nn.relu(x)
        x = self.pool(x)
        
        return x


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels//8, 1, kernel_size=1)
    
    def __call__(self, x):
        # Generate attention weights
        attn = nn.relu(self.conv1(x))
        attn = mx.sigmoid(self.conv2(attn))
        
        # Apply attention weights
        return x * attn


def load_lfw_dataset(min_faces_per_person=10, resize=True):
    """Load the LFW dataset and filter people with a minimum number of faces."""
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


def preprocess_image(img):
    """Preprocess PIL Image for the model."""
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


def generate_triplets(identities, num_triplets=1000):
    """Generate triplets for training: (anchor, positive, negative)."""
    triplets = []
    
    # Filter identities with at least 2 images
    valid_identities = {k: v for k, v in identities.items() if len(v) >= 2}
    identity_list = list(valid_identities.keys())
    
    if len(identity_list) < 2:
        raise ValueError("Need at least 2 different identities for triplet generation")
    
    for _ in range(num_triplets):
        # Select anchor identity
        anchor_id = random.choice(identity_list)
        
        # Select negative identity (different from anchor)
        negative_ids = [id for id in identity_list if id != anchor_id]
        negative_id = random.choice(negative_ids)
        
        # Select anchor and positive (same identity)
        anchor_positive_imgs = random.sample(valid_identities[anchor_id], 2)
        anchor_img = anchor_positive_imgs[0][0]  # Get the image from the tuple
        positive_img = anchor_positive_imgs[1][0]
        
        # Select negative (different identity)
        negative_img = random.choice(valid_identities[negative_id])[0]
        
        # Add to triplets
        triplets.append((anchor_img, positive_img, negative_img))
    
    return triplets


def create_batch_triplets(triplets, batch_size=32):
    """Create batches of triplets for training."""
    batches = []
    
    for i in range(0, len(triplets), batch_size):
        batch_triplets = triplets[i:i+batch_size]
        
        # Load and preprocess images
        anchor_batch = []
        positive_batch = []
        negative_batch = []
        
        for anchor, positive, negative in batch_triplets:
            anchor_batch.append(preprocess_image(anchor))
            positive_batch.append(preprocess_image(positive))
            negative_batch.append(preprocess_image(negative))
        
        # Stack arrays
        anchor_batch = mx.stack(anchor_batch)
        positive_batch = mx.stack(positive_batch)
        negative_batch = mx.stack(negative_batch)
        
        batches.append((anchor_batch, positive_batch, negative_batch))
    
    return batches


def triplet_loss(anchor, positive, negative, margin=MARGIN):
    """Compute triplet loss."""
    pos_dist = mx.sum(mx.square(anchor - positive), axis=-1)
    neg_dist = mx.sum(mx.square(anchor - negative), axis=-1)
    return mx.mean(mx.maximum(pos_dist - neg_dist + margin, 0))


def train_model(model, triplet_batches, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    """Train the model with triplet loss using batches."""
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    # Define loss function for batched inputs
    def compute_loss(params, batch):
        """Compute loss given model parameters and batch of triplets."""
        anchor_batch, positive_batch, negative_batch = batch
        
        # Update model parameters
        model.update(params)
        
        # Get embeddings
        anchor_emb = model(anchor_batch)
        pos_emb = model(positive_batch)
        neg_emb = model(negative_batch)
        
        return triplet_loss(anchor_emb, pos_emb, neg_emb)
    
    # Get initial model parameters
    params = model.parameters()
    
    for epoch in range(epochs):
        random.shuffle(triplet_batches)
        epoch_loss = 0
        
        progress_bar = tqdm(triplet_batches, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            # Compute value and gradients with respect to model parameters
            loss, grads = mx.value_and_grad(compute_loss)(params, batch)
            
            # Update parameters
            updates, optimizer.state = optimizer.update(grads, optimizer.state, params)
            params = optim.apply_updates(params, updates)
            model.update(params)
            
            # Track loss
            loss_value = loss.item()
            epoch_loss += loss_value
            progress_bar.set_postfix({"batch_loss": f"{loss_value:.4f}"})
        
        avg_loss = epoch_loss / len(triplet_batches)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return model


def evaluate_model(model, test_triplets, batch_size=32):
    """Evaluate model performance on test triplets."""
    correct = 0
    total = len(test_triplets)
    
    # Process in batches
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
        
        # Get embeddings
        anchor_emb = model(anchor_batch)
        pos_emb = model(positive_batch)
        neg_emb = model(negative_batch)
        
        # Calculate distances
        pos_dist = mx.sum(mx.square(anchor_emb - pos_emb), axis=-1)
        neg_dist = mx.sum(mx.square(anchor_emb - neg_emb), axis=-1)
        
        # Count correct predictions (positive should be closer than negative)
        batch_correct = mx.sum(pos_dist < neg_dist).item()
        correct += batch_correct
    
    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def save_model(model, path="lfw_face_recognition_model.npz"):
    """Save model weights."""
    weights = {k: v.numpy() for k, v in model.parameters().items()}
    np.savez(path, **weights)
    print(f"Model saved to {path}")


def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # 1. Load LFW dataset
    identities, lfw_people = load_lfw_dataset(min_faces_per_person=20)
    print(f"Dataset contains {len(identities)} people with sufficient images")
    
    # 2. Generate triplets
    num_triplets = 5000
    print(f"Generating {num_triplets} triplets...")
    all_triplets = generate_triplets(identities, num_triplets=num_triplets)
    
    # 3. Split into train and test sets
    train_triplets, test_triplets = train_test_split(all_triplets, test_size=0.2, random_state=42)
    print(f"Split data into {len(train_triplets)} training and {len(test_triplets)} testing triplets")
    
    # 4. Create batches for training
    print("Creating training batches...")
    train_batches = create_batch_triplets(train_triplets, batch_size=BATCH_SIZE)
    
    # 5. Initialize model
    print("Initializing model...")
    model = ImprovedFaceCNN(embedding_dim=512)
    
    # 6. Train model
    print("Starting training...")
    trained_model = train_model(model, train_batches, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    
    # 7. Evaluate model
    print("Evaluating model...")
    evaluate_model(trained_model, test_triplets)
    
    # 8. Save model
    save_model(trained_model)
    
    print("Training complete!")


if __name__ == "__main__":
    main()