"""实测反射系数条件 Flow; measured input (N,2), normalized output lengths."""

from networks import build

MODEL_NAME = "conditional_flow"


def Model(width=128, blocks=4, layers=8, fourier_features=32, fourier_scale=1.0):
    return build(
        MODEL_NAME,
        dict(
            width=width,
            blocks=blocks,
            layers=layers,
            fourier_features=fourier_features,
            fourier_scale=fourier_scale,
        ),
    )
