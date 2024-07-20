import functools


def _handle_exceptions(func):
    """
    Method for function arguments validation.

    NOTE: Not to be called directly.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            if func.__name__ == "nstep_distribution":
                n_steps = args[1]
                if not isinstance(n_steps, int) or n_steps < 0:
                    raise ValueError(
                        "Argument 'n_steps' must be a non-negative integer."
                    )

            elif func.__name__ in ["is_communicating", "is_transient", "is_recurrent"]:
                states = args[0].states
                for state in args[1:]:
                    if not isinstance(state, str) or state not in states:
                        raise ValueError(
                            "Argument 'state' must be a string and present in MarkovChain.states."
                        )

            elif func.__name__ in ["fit"]:
                if len(args) == 2:
                    data = args[1]
                else:
                    data = kwargs.get("data")

                if (
                    not (isinstance(data, str) or isinstance(data, list))
                    or len(data) == 0
                ):
                    raise ValueError("Argument 'data' must be a non-empty string.")

            elif func.__name__ == "simulate":
                if len(args) > 2:
                    n_steps = args[2]
                else:
                    n_steps = kwargs.get("n_steps")

                if len(args) > 1:
                    initial_state = args[1]
                else:
                    initial_state = kwargs.get("initial_state")

                if not isinstance(n_steps, int) or n_steps <= 0:
                    raise ValueError("Argument 'n_steps' must be a non-empty string.")

                if not isinstance(initial_state, str) or (
                    initial_state not in args[0].states
                ):
                    raise ValueError(
                        "Argument 'initial_state' must be a non-empty string present \
                                    in `MarkovChain.states`."
                    )

            elif func.__name__ == "save_model":
                path = args[1]
                if not isinstance(path, str) or not path.endswith(".json"):
                    raise ValueError(
                        "Argument 'path' must be a valid filepath ending with '.json'."
                    )

            return func(*args, **kwargs)

        except Exception as e:
            print(f"Exception in function '{func.__name__}': {str(e)}")
            raise

    return wrapper


class _ChainopyException(Exception):
    # Not Implemented
    pass
