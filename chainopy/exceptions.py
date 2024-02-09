import functools
import os


def handle_exceptions(func):
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

            elif func.__name__ == "fit":
                data = args[1]
                if not isinstance(data, str) or len(data) == 0:
                    raise ValueError("Argument 'data' must be a non-empty string.")

            elif func.__name__ in ["save_model", "load_model"]:
                path = args[1]
                if (
                    not isinstance(path, str)
                    or not os.path.isfile(path)
                    or not path.endswith(".json")
                ):
                    raise ValueError(
                        "Argument 'path' must be a valid filepath ending with '.json'."
                    )

            return func(*args, **kwargs)

        except Exception as e:
            print(f"Exception in function '{func.__name__}': {str(e)}")
            raise

    return wrapper
