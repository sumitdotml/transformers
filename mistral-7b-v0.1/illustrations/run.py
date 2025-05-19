#!/usr/bin/env python3
import argparse
import os
import sys
import importlib.util

# Get the absolute path of the current file's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get parent directory to access modules
PARENT_DIR = os.path.dirname(CURRENT_DIR)


# Function to import a module from a file path
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not find module {module_name} at {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_rolling_buffer_visualization():
    """Run the rolling buffer cache visualization."""
    print("\n=== RUNNING ROLLING BUFFER CACHE VISUALIZATION ===\n")

    # Import the visualization module
    module_path = os.path.join(CURRENT_DIR, "_visualize_rolling_buffer_plot.py")
    try:
        # Temporarily add parent directory to path for imports
        sys.path.insert(0, PARENT_DIR)
        module = import_module_from_file("rolling_buffer_viz", module_path)
        # The module runs the visualization when imported
        print(f"Rolling buffer visualization completed successfully")
    except Exception as e:
        print(f"Error running rolling buffer visualization: {e}")
    finally:
        # Remove parent directory from path
        if PARENT_DIR in sys.path:
            sys.path.remove(PARENT_DIR)


def run_sliding_window_visualization():
    """Run the sliding window evolution visualization."""
    print("\n=== RUNNING SLIDING WINDOW EVOLUTION VISUALIZATION ===\n")

    # Import the visualization module
    module_path = os.path.join(CURRENT_DIR, "_visualize_sliding_window.py")
    try:
        # Temporarily add parent directory to path for imports
        sys.path.insert(0, PARENT_DIR)
        module = import_module_from_file("sliding_window_viz", module_path)
        # The module runs the visualization when imported
        print(f"Sliding window evolution visualization completed successfully")
    except Exception as e:
        print(f"Error running sliding window evolution visualization: {e}")
    finally:
        # Remove parent directory from path
        if PARENT_DIR in sys.path:
            sys.path.remove(PARENT_DIR)


def run_sliding_window_attention():
    """Run the sliding window attention visualization."""
    print("\n=== RUNNING SLIDING WINDOW ATTENTION VISUALIZATION ===\n")

    # Import the visualization module
    module_path = os.path.join(CURRENT_DIR, "_visualize_swa.py")
    try:
        # Temporarily add parent directory to path for imports
        sys.path.insert(0, PARENT_DIR)
        module = import_module_from_file("swa_viz", module_path)
        # The module runs the visualization when imported
        print(f"Sliding window attention visualization completed successfully")
    except Exception as e:
        print(f"Error running sliding window attention visualization: {e}")
    finally:
        # Remove parent directory from path
        if PARENT_DIR in sys.path:
            sys.path.remove(PARENT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Generate Mistral visualizations")
    parser.add_argument(
        "--all", action="store_true", help="Generate all visualizations"
    )
    parser.add_argument(
        "--rolling-buffer",
        action="store_true",
        help="Generate rolling buffer visualization",
    )
    parser.add_argument(
        "--sliding-window",
        action="store_true",
        help="Generate sliding window evolution visualization",
    )
    parser.add_argument(
        "--sliding-window-attention",
        action="store_true",
        help="Generate sliding window attention visualization",
    )

    args = parser.parse_args()

    # If no arguments provided, show help
    if not (
        args.all
        or args.rolling_buffer
        or args.sliding_window
        or args.sliding_window_attention
    ):
        parser.print_help()
        return

    try:
        # Run selected visualizations
        if args.all or args.rolling_buffer:
            run_rolling_buffer_visualization()

        if args.all or args.sliding_window:
            run_sliding_window_visualization()

        if args.all or args.sliding_window_attention:
            run_sliding_window_attention()

        print("\nAll requested visualizations completed successfully!")
    except Exception as e:
        print(f"Error generating visualizations: {e}")


if __name__ == "__main__":
    main()
