from model.transformer import GeneratorTransformer
from tokenizers import Tokenizer
import torch 
from data.dataset import ZaratustraDataset, create_dataloaders


def chat():
    config = {
        'batch_size': 1,
        'max_length': 128,
        'max_train_samples': 5000000,
        'max_val_samples': 3000,
        'd_model': 256,
        'nhead': 8,
        'num_decoder_layers': 4,
        'd_ff': 1024,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'num_epochs': 4,
        'save_epochs': 4,
        'save_dir': 'checkpoints',
    }
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    train_loader, val_loader, dataset = create_dataloaders(
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        max_train_samples=config['max_train_samples'],
        max_val_samples=config['max_val_samples'],
    )
    checkpoint = torch.load("checkpoints/epoch_2.pt", map_location='cuda')
    model = GeneratorTransformer(
        d_model=config['d_model'],
        num_heads=config['nhead'],
        d_ff=config['d_ff'],
        num_layers=config['num_decoder_layers'],
        vocab_size=dataset.get_vocab_size(),
        pad_index=dataset.get_pad_token_id(),
        dropout=config['dropout'],
        tokenizer=dataset.tokenizer,
        max_len=config['max_length'],
    ).to('cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'quit':
            break
            
        response = model.generate(user_input, context_len=50, max_out_tokens=200, temperature=0.8)
        print(f"Бот: {response}")

if __name__ == "__main__":
    chat()