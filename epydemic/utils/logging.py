def build_logger(verbose, prefix="default"):
    def _silent_log(*s):
        pass

    def _log(*s):
        print(f"{prefix}:", *s)

    return _log if verbose else _silent_log