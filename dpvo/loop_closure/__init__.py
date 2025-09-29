# Conditional imports for loop closure functionality
# These require compiled CUDA extensions that may not be available

# Initialize availability flags
OPTIM_UTILS_AVAILABLE = False
LONG_TERM_AVAILABLE = False

try:
    from .optim_utils import *
    OPTIM_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Loop closure optim_utils not available: {e}")
except Exception as e:
    print(f"Warning: Loop closure optim_utils failed to load: {e}")

try:
    from .long_term import *
    LONG_TERM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Loop closure long_term not available: {e}")
except Exception as e:
    print(f"Warning: Loop closure long_term failed to load: {e}")

# Export availability flags for users to check
__all__ = ['OPTIM_UTILS_AVAILABLE', 'LONG_TERM_AVAILABLE']
