import abc
import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pass

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        pass


class AutoregressiveModel(torch.nn.Module, Autoregressive):

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens

        # 토큰 임베딩: 정수 토큰 → d_latent 차원 벡터
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)

        # 시작 토큰 (learnable start token) - 시퀀스 시프트용
        self.start_token = torch.nn.Parameter(torch.zeros(1, 1, d_latent))

        # Transformer (decoder-only)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,   # (B, seq, dim) 순서 사용
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 출력: d_latent → n_tokens (다음 토큰 확률)
        self.output_proj = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: (B, h, w) 정수 토큰
        return: (B, h, w, n_tokens) 다음 토큰 확률
        """
        B, h, w = x.shape
        seq_len = h * w

        # 1. flatten: (B, h, w) → (B, seq_len)
        x = x.reshape(B, seq_len)

        # 2. 임베딩: (B, seq_len) → (B, seq_len, d_latent)
        x = self.embedding(x)

        # 3. 시프트: start_token 앞에 붙이고 마지막 토큰 제거
        #    [start, token0, token1, ..., token(N-2)]
        #    → 출력[i]가 입력[i]를 예측하게 됨
        start = self.start_token.expand(B, 1, self.d_latent)
        x = torch.cat([start, x[:, :-1, :]], dim=1)  # (B, seq_len, d_latent)

        # 4. causal mask: 미래 토큰 못 보게
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # 5. Transformer
        x = self.transformer(x, mask=mask, is_causal=True)  # (B, seq_len, d_latent)

        # 6. 출력 projection
        x = self.output_proj(x)  # (B, seq_len, n_tokens)

        # 7. reshape: (B, seq_len, n_tokens) → (B, h, w, n_tokens)
        x = x.reshape(B, h, w, self.n_tokens)

        return x, {}

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        """
        토큰을 하나씩 auto-regressive하게 생성
        return: (B, h, w) 정수 토큰
        """
        seq_len = h * w
        tokens = torch.zeros(B, 0, dtype=torch.long, device=device)

        with torch.no_grad():
            for i in range(seq_len):
                # 현재까지의 토큰으로 다음 토큰 예측
                if tokens.shape[1] == 0:
                    # 첫번째 토큰: start_token만 있는 상태
                    x = self.start_token.expand(B, 1, self.d_latent)
                else:
                    # 임베딩 후 start_token 붙이기
                    emb = self.embedding(tokens)              # (B, i, d_latent)
                    start = self.start_token.expand(B, 1, self.d_latent)
                    x = torch.cat([start, emb], dim=1)        # (B, i+1, d_latent)

                # causal mask
                mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=device)

                # Transformer
                out = self.transformer(x, mask=mask, is_causal=True)  # (B, i+1, d_latent)

                # 마지막 위치의 logit으로 다음 토큰 샘플링
                logits = self.output_proj(out[:, -1, :])      # (B, n_tokens)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)      # (B, 1)

                tokens = torch.cat([tokens, next_token], dim=1)  # (B, i+1)

        return tokens.reshape(B, h, w)