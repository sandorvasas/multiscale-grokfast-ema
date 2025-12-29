# Multi-scale Grok Fast with Exponential Moving Average

Based on: https://arxiv.org/pdf/2405.20233 \

### About GrokFast EMA
Standard GrokFast uses a moving average of the last 100 or so gradients, but that bloats memory af, so I made an a version of it using an exponential moving average based approximation of it, so it's O(1) and only stores a single copy of the gradients, like the momentum term in Adam. \

### About Multi-scale Grok Fast EMA
Then the question arose in me, why choose one window size, when we can have multiple? Just like when we trade stocks, we need more EMAs. Then the EMAs are averaged. This could potentially be improved by dynamically assigning weights to each GrokFast and play around with that WMA of Groks.

## How to use

Put it in your optimizer, e.g.

```py
class MyOwnOptimizerLikeAdamW(Optimizer):
    def __init__(
        self,
        params,
        ...whatever...
        grok_fast = [(100, 2), (50, 1), (25, 0.5), (10, 0.2)]
    ):
        ...stuff here...

        super().__init__(params, defaults)

        ...stuff here too...

        self.grok_fast = MultiGrokFastEMA(self.param_groups, grok_fast)

    @torch.no_grad()
    def step(self, training_step: int):
        self.grok_fast.step()

        ...the usual optimizer.step code here...

``
