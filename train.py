import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader

def train_model():
    # Load Dataset from Hugging face Dataset library
    full_dataset = load_dataset("stanfordnlp/imdb", split="train")
    dataset = full_dataset.shuffle(seed=42).select(range(3000))

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Optimizer for parameter
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train
    model.train()

    # Create a dataloader for batching (for better training speed and efficiency)
    dataloaders = DataLoader (dataset, batch_size=8, shuffle=True)

    # Training Loop
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloaders:
            inputs = tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
            labels = torch.tensor(batch["label"]).to(device)

            optimizer.zero_grad()
            # ** for unpacking I/P Dict
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloaders)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save the model
    model.save_pretrained("./model/")
    tokenizer.save_pretrained("./model/")

    # Test the model with sample sentences
    test_sentences = [
        "Absolutely loved every minute of it!",
        "One of the worst movies I've ever seen.",
        "It had its moments, but overall it fell flat.",
        "Incredible visuals and a powerful story.",
        "I regret wasting my time on this.",
        "A truly unforgettable performance by the lead actor.",
        "Nothing special, just another average film.",
        "The plot was weak and the pacing was off.",
        "Heartwarming, funny, and full of charm.",
        "I couldnt stay awake — it was that boring."
    ]

    # Eval mode
    model.eval()

    # Tokenizer for test input
    inputs = tokenizer(test_sentences, truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)

    # Disable Backpropagation and Gradient Descent
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    # Print Predictions
    for sentence, prediction in zip(test_sentences, predictions):
        sentiment = "positive" if prediction.item() == 1 else "negative"
        print(f"Input: \"{sentence}\" -> Predicted sentiment: {sentiment}")

# Call for train and test
train_model()
    