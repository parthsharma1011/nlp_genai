import tensorflow as tf
import numpy as np
from collections import Counter
import random
#
# Check TensorFlow and Keras versions for compatibility
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
if not tf.__version__.startswith('2'):
    raise ValueError("This code requires TensorFlow 2.x")

# Step 1: Prepare the dataset #main khush hun
data = [
    ("I am happy", "मैं खुश हूँ"),
    ("You are sad", "तुम उदास हो"),
    ("She is tired", "वह थकी हुई है"),
    ("We are hungry", "हम भूखे हैं"),
    ("He is angry", "वह गुस्से में है"),
    ("They are busy", "वे व्यस्त हैं"),
    ("I am cold", "मुझे ठंड लग रही है"),
    ("You are late", "तुम देर से हो"),
    ("She is happy", "वह खुश है"),
    ("We are ready", "हम तैयार हैं")
]

# Build vocabularies
def build_vocab(sentences, lang):
    tokens = Counter()
    for sent in sentences:
        tokens.update(sent.split())
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    for i, token in enumerate(tokens.keys(), 3):
        vocab[token] = i
    return vocab

# Extract English and Hindi sentences
eng_sents = [pair[0] for pair in data]
hin_sents = [pair[1] for pair in data]
eng_vocab = build_vocab(eng_sents, "eng")
hin_vocab = build_vocab(hin_sents, "hin")

print(f"English vocabulary size: {len(eng_vocab)}")
print(f"Hindi vocabulary size: {len(hin_vocab)}")

# Convert sentences to indices
def sentence_to_indices(sent, vocab):
    indices = [vocab.get(token, vocab.get("<UNK>", 0)) for token in sent.split()]
    indices = [vocab["<SOS>"]] + indices + [vocab["<EOS>"]]
    return indices

# Prepare padded data
def prepare_data(data, eng_vocab, hin_vocab):
    src_data = [sentence_to_indices(pair[0], eng_vocab) for pair in data]
    tgt_data = [sentence_to_indices(pair[1], hin_vocab) for pair in data]
    # Pad sequences
    src_padded = tf.keras.preprocessing.sequence.pad_sequences(
        src_data, padding='post', value=eng_vocab["<PAD>"])
    tgt_padded = tf.keras.preprocessing.sequence.pad_sequences(
        tgt_data, padding='post', value=hin_vocab["<PAD>"])
    return src_padded, tgt_padded

src_data, tgt_data = prepare_data(data, eng_vocab, hin_vocab)

print(f"Source data shape: {src_data.shape}")
print(f"Target data shape: {tgt_data.shape}")

# Step 2: Define the Encoder
class Encoder(tf.keras.Model):
    def __init__(self, input_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_size, embed_size)
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_state=True)

    def call(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_size]
        _, hidden, cell = self.lstm(embedded)  # hidden, cell: [batch_size, hidden_size]
        return hidden, cell

# Step 3: Define the Decoder
class Decoder(tf.keras.Model):
    def __init__(self, output_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(output_size, embed_size)
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(output_size)

    def call(self, x, hidden, cell):
        embedded = self.embedding(x)  # [batch_size, 1, embed_size]
        lstm_out, hidden, cell = self.lstm(embedded, initial_state=[hidden, cell])
        output = self.fc(lstm_out)  # [batch_size, 1, output_size]
        return output, hidden, cell

# Step 4: Define the Seq2Seq Model
class Seq2Seq(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False, teacher_forcing_ratio=0.5):
        src, tgt = inputs
        batch_size = tf.shape(src)[0]
        tgt_len = tf.shape(tgt)[1]
        tgt_vocab_size = len(hin_vocab)

        # Encode
        hidden, cell = self.encoder(src)

        # Initialize outputs tensor
        outputs = []

        # Start with <SOS>
        input_token = tgt[:, 0:1]  # [batch_size, 1]

        # Decode step by step
        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs.append(output)

            if training:
                teacher_force = random.random() < teacher_forcing_ratio
                input_token = tgt[:, t:t+1] if teacher_force else tf.argmax(output, axis=-1, output_type=tf.int32)
            else:
                input_token = tf.argmax(output, axis=-1, output_type=tf.int32)

        # Concatenate all outputs
        if outputs:
            return tf.concat(outputs, axis=1)
        else:
            # Return zeros if no outputs
            return tf.zeros((batch_size, 1, tgt_vocab_size))

# Step 5: Training
def train(model, src_data, tgt_data, epochs=100):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    # Fix: Use the correct loss function for newer Keras versions
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def train_step(src, tgt):
        with tf.GradientTape() as tape:
            outputs = model([src, tgt], training=True, teacher_forcing_ratio=1.0)

            # Get target labels (excluding the first SOS token)
            target_labels = tgt[:, 1:]

            # Ensure outputs match the target sequence length
            seq_len = tf.shape(target_labels)[1]
            outputs = outputs[:, :seq_len, :]

            # Create mask to ignore padding tokens
            mask = tf.cast(target_labels != hin_vocab["<PAD>"], tf.float32)

            # Calculate loss
            loss = loss_fn(target_labels, outputs)
            loss = loss * mask

            # Calculate mean loss (only over non-padded tokens)
            total_loss = tf.reduce_sum(loss)
            total_tokens = tf.reduce_sum(mask)
            mean_loss = total_loss / (total_tokens + 1e-8)  # Add small epsilon to avoid division by zero

        gradients = tape.gradient(mean_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mean_loss

    print("Starting training...")
    for epoch in range(epochs):
        loss = train_step(src_data, tgt_data)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

    print("Training completed!")

# Initialize model
input_size = len(eng_vocab)
output_size = len(hin_vocab)
embed_size = 50
hidden_size = 100

print(f"\nInitializing model with:")
print(f"Input vocab size: {input_size}")
print(f"Output vocab size: {output_size}")
print(f"Embedding size: {embed_size}")
print(f"Hidden size: {hidden_size}")

encoder = Encoder(input_size, embed_size, hidden_size)
decoder = Decoder(output_size, embed_size, hidden_size)
model = Seq2Seq(encoder, decoder)

# Build the model by calling it once
dummy_src = tf.zeros((1, 5), dtype=tf.int32)
dummy_tgt = tf.zeros((1, 5), dtype=tf.int32)
_ = model([dummy_src, dummy_tgt], training=False)

print(f"Model built successfully!")
print(f"Total trainable parameters: {sum([tf.size(var).numpy() for var in model.trainable_variables])}")

# Train the model
train(model, src_data, tgt_data, epochs=50)

# Step 6: Inference
def translate(model, sentence, eng_vocab, hin_vocab, max_len=15):
    """Translate an English sentence to Hindi"""
    print(f"\nTranslating: '{sentence}'")

    # Tokenize and convert to indices
    tokens = sentence.split()
    indices = [eng_vocab["<SOS>"]] + [eng_vocab.get(token, 0) for token in tokens] + [eng_vocab["<EOS>"]]
    print(f"Input tokens: {tokens}")
    print(f"Input indices: {indices}")

    # Convert to tensor and add batch dimension
    src_tensor = tf.convert_to_tensor([indices], dtype=tf.int32)

    # Encode
    hidden, cell = model.encoder(src_tensor)
    print(f"Encoded to context vector of shape: {hidden.shape}")

    # Decode step by step
    input_token = tf.convert_to_tensor([[hin_vocab["<SOS>"]]], dtype=tf.int32)
    output_tokens = []

    print("Decoding steps:")
    for step in range(max_len):
        output, hidden, cell = model.decoder(input_token, hidden, cell)
        predicted_token = tf.argmax(output, axis=-1).numpy()[0, 0]

        # Get the word for this token
        inv_hin_vocab = {v: k for k, v in hin_vocab.items()}
        predicted_word = inv_hin_vocab.get(predicted_token, "<UNK>")

        print(f"  Step {step+1}: {predicted_word} (token {predicted_token})")

        if predicted_token == hin_vocab["<EOS>"]:
            print("  Reached EOS token, stopping")
            break

        output_tokens.append(predicted_token)
        input_token = tf.convert_to_tensor([[predicted_token]], dtype=tf.int32)

    # Convert indices to words
    inv_hin_vocab = {v: k for k, v in hin_vocab.items()}
    translated_words = [inv_hin_vocab.get(idx, "<UNK>") for idx in output_tokens]
    translation = " ".join(translated_words)

    print(f"Final translation: '{translation}'")
    return translation

# Test translations
print("\n" + "="*50)
print("TESTING TRANSLATIONS")
print("="*50)

# Test on training data
test_sentences = ["I am happy", "You are sad", "She is tired"]
for sentence in test_sentences:
    # Find expected translation
    expected = None
    for eng, hin in data:
        if eng == sentence:
            expected = hin
            break

    translation = translate(model, sentence, eng_vocab, hin_vocab)
    print(f"Expected: '{expected}'")
    print(f"Got:      '{translation}'")
    print("-" * 30)

# # Test on new sentences
# print("\nTesting on NEW sentences (not in training data):")
# new_sentences = ["I am ready", "You are busy", "He is late"]
# for sentence in new_sentences:
#     translation = translate(model, sentence, eng_vocab, hin_vocab)
#     print("-" * 30)

# print("\nTranslation testing completed!")

# # Show vocabulary mappings for reference
# print("\n" + "="*50)
# print("VOCABULARY REFERENCE")
# print("="*50)
# print("English vocabulary:")
# for word, idx in list(eng_vocab.items())[:10]:
#     print(f"  {word}: {idx}")

# print("\nHindi vocabulary:")
# for word, idx in list(hin_vocab.items())[:10]:
#     print(f"  {word}: {idx}")