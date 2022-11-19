class EmptyText(Exception):
    def __init__(
        self,
        message="The input text is empty",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(message, *args, **kwargs)
