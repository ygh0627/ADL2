import abc
import torch
from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path
    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self._embedding_dim = embedding_dim

        # down-projection: embedding_dim → codebook_bits
        self.down_proj = torch.nn.Linear(embedding_dim, codebook_bits, bias=False)
        # up-projection: codebook_bits → embedding_dim
        self.up_proj = torch.nn.Linear(codebook_bits, embedding_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., embedding_dim)
        return: (..., codebook_bits)  각 값은 -1 or 1 (differentiable)
        """
        z = self.down_proj(x)                     # (..., codebook_bits)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)  # L2 정규화
        z = diff_sign(z)                           # binarize: -1 or 1
        return z

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., codebook_bits)
        return: (..., embedding_dim)
        """
        return self.up_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * (2 ** torch.arange(x.size(-1)).to(x.device))).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        # PatchAutoEncoder의 bottleneck = latent_dim (BSQ가 따로 차원 축소함)
        super().__init__(patch_size=patch_size, latent_dim=latent_dim, bottleneck=latent_dim)
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)
        self.codebook_bits = codebook_bits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        이미지 → AE encoder → BSQ encode
        x: (B, H, W, 3)
        return: (B, h, w, codebook_bits)  값은 -1 or 1
        """
        z = super().encode(x)       # (B, h, w, latent_dim)  ← PatchAutoEncoder.encode
        return self.bsq.encode(z)   # (B, h, w, codebook_bits)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        BSQ code → BSQ decode → AE decoder → 이미지
        x: (B, h, w, codebook_bits)
        return: (B, H, W, 3)
        """
        z = self.bsq.decode(x)      # (B, h, w, latent_dim)
        return super().decode(z)    # (B, H, W, 3)  ← PatchAutoEncoder.decode

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, 3)
        return: (B, h, w)  정수 토큰
        """
        z = super().encode(x)             # (B, h, w, latent_dim)
        return self.bsq.encode_index(z)   # (B, h, w)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, h, w)  정수 토큰
        return: (B, H, W, 3)
        """
        z = self.bsq.decode_index(x)   # (B, h, w, latent_dim)
        return super().decode(z)        # (B, H, W, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = super().encode(x)           # (B, h, w, latent_dim)
        z_q = self.bsq(z)               # (B, h, w, latent_dim)  quantize → dequantize
        x_hat = super().decode(z_q)     # (B, H, W, 3)

        # 코드북 사용률 모니터링
        with torch.no_grad():
            cnt = torch.bincount(
                self.encode_index(x).flatten(),
                minlength=2 ** self.codebook_bits
            )

        return x_hat, {
            "cb0": (cnt == 0).float().mean(),   # 한번도 안쓰인 코드 비율
            "cb2": (cnt <= 2).float().mean(),   # 2번 이하 쓰인 코드 비율
        }