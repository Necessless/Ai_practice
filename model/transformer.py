import torch
from tokenizers import Tokenizer 
from model.transformer_layers import MultiheadAttention, FeedForward, Decoder, DecoderLayer, Embedding


def get_pad_mask(x: torch.Tensor, pad_index: int):
    return (x != pad_index).unsqueeze(-2)


def get_subsequent_mask(x: torch.Tensor):
    batch_size, seq_len = x.size()
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class GeneratorTransformer(torch.nn.Module):

    def __init__(
            self,
            d_model: int = 64,
            num_heads: int = 8,
            d_ff: int = 512,
            num_layers: int = 6,
            vocab_size: int = 1000,
            pad_index: int = 1,
            dropout: float = 0.1,
            max_len: int = 64,
            tokenizer: Tokenizer = None,
            device: str = 'cuda'
    ):
        super().__init__()
        mha = MultiheadAttention(d_model, num_heads)
        ffn = FeedForward(d_model, d_ff)
        self.decoder = Decoder(DecoderLayer(mha, ffn, dropout), num_layers)
        self.normalize = torch.nn.LayerNorm(d_model)
        self.tgt_embedding = Embedding(d_model, vocab_size, pad_index)
        self.vocab_projection = torch.nn.Linear(d_model, vocab_size)

        self.pad_index = pad_index
        self.device = device
        self.max_len = max_len
        self.tokenizer = tokenizer

    def decode_tgt(self, x) -> torch.Tensor:
        tgt_mask = get_pad_mask(x, self.pad_index) & get_subsequent_mask(x).to(self.device)
        x = self.tgt_embedding(x)
        x = self.decoder.forward(x, tgt_mask)
        x = self.normalize(x)
        return self.vocab_projection(x)

    def forward(self, x):
        out = self.decode_tgt(x)
        return out

    @staticmethod
    def load_model(checkpoint_path: str, tokenizer):
        model = GeneratorTransformer(tokenizer=tokenizer)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.to(device)
        model.load_state_dict(state_dict)
        return model

    def generate(self, prompt, context_len=50, temperature=1.0, max_out_tokens=200):
        """
        Генерирует ответ на основе промпта.

        При авторегрессии контекст сдвигается на 1 токен влево:
        - Изначально: [prompt_tokens]
        - После первого предсказания: [prompt_tokens, predicted_token]
        - При следующем предсказании: [prompt_tokens[1:], predicted_token, new_prediction]
        - И так далее, пока не достигнем max_length или EOS
        """
        self.eval()
        eos_token_id = self.tokenizer.token_to_id('</s>') 
        with torch.no_grad():
            # Токенизируйте промпт
            input_ids = self.tokenizer.encode(prompt).ids
            input_ids = torch.tensor([input_ids]).to(self.device)

            generated = input_ids.clone()

            for _ in range(max_out_tokens):
                # Получите предсказание для последнего токена
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature

                # Выберите следующий токен
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)

                # Добавьте к результату
                generated = torch.cat([generated, next_token], dim=1)

                # Сдвиньте контекст на 1 токен влево для следующей итерации
                input_ids = generated[:, -context_len:]

                # Проверьте на EOS
                if next_token.item() == eos_token_id:
                    break

        return self.tokenizer.decode(generated[0].tolist())

    def generate_from_ids(self, input_ids, context_len=50, temperature=1.0, max_out_tokens=200):
        """
        Генерирует ответ на основе промпта.

        При авторегрессии контекст сдвигается на 1 токен влево:
        - Изначально: [prompt_tokens]
        - После первого предсказания: [prompt_tokens, predicted_token]
        - При следующем предсказании: [prompt_tokens[1:], predicted_token, new_prediction]
        - И так далее, пока не достигнем max_length или EOS
        """
        self.eval()
        eos_token_id = self.tokenizer.token_to_id('</s>') 
        with torch.no_grad():
            generated = input_ids.clone()

            for _ in range(max_out_tokens):
                # Получите предсказание для последнего токена
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature

                # Выберите следующий токен
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)

                # Добавьте к результату
                generated = torch.cat([generated, next_token], dim=1)

                # Сдвиньте контекст на 1 токен влево для следующей итерации
                input_ids = generated[:, -context_len:]

                # Проверьте на EOS
                if next_token.item() == eos_token_id:
                    break

        return generated
