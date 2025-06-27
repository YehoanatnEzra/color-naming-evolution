import encoder_strategies


class Language:
    """
    Represents a language with its encoder, lexicon size, and generation strategy.
    """
    def __init__(self, num_meanings, lexicon_size, strategy=0, strategy_params=None):
        self.lexicon_size = lexicon_size
        self.strategy = strategy
        self.strategy_params = strategy_params or {}
        self.encoder = encoder_strategies.get_encoder_by_strategy(strategy, num_meanings, lexicon_size, **self.strategy_params)