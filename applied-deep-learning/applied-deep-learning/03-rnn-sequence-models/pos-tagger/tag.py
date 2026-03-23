import torch
import pickle
from models.PoSGRU import PoSGRU

def main():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    checkpoint = torch.load('best_model.pt', map_location='cpu')
    cfg = checkpoint['config']

    model = PoSGRU(
        vocab_size=vocab.lenWords(),
        embed_dim=cfg['embed_dim'],
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['layers'],
        output_dim=vocab.lenLabels(),
        residual=cfg['residual']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Pos Tagger")

    while True:
        sentence = input("\nEnter a sentence or quit: ")
        if sentence.strip().lower() == 'quit':
            break

        tokens = sentence.strip().split()
        tokens_lower = [t.lower() for t in tokens]
        word_ids = vocab.numeralizeSentence(tokens_lower)
        x = torch.LongTensor(word_ids).unsqueeze(0)

        with torch.no_grad():
            out = model(x)
            preds = torch.argmax(out, dim=-1).squeeze(0).tolist()

        tags = vocab.denumeralizeLabels(preds)
        print()
        for tok, tag in zip(tokens, tags):
            print(tok, "-", tag)

if __name__ == "__main__":
    main()