import os
import json

class UsefulTools:
    """
        useful_tools.py
        ===============

        A general-purpose utility module that provides commonly needed tools across data science,
        machine learning, and application development projects.

        Currently implemented utilities:
        - JsonCache: A static class for caching JSON-serializable data to disk with validation.

        Planned extensions:
        - TextCleaner: Utilities for normalizing and preprocessing raw text data
        - Timer: Context manager for measuring execution time
        - Logger: Lightweight, customizable logging wrapper
        - PlotHelper: Easy-to-use plotting wrappers for common visualizations

        Classes
        -------

        class UsefulTools.JsonCache
        ---------------------------
        Static methods to cache data to and load data from local JSON files. Designed for flexible use
        across various components of a project that benefit from caching (e.g., API responses, model results, etc.).

        Methods:
            - load(filename: str, expected_type: type = list) -> Optional[Any]
                Loads JSON content from a file if it exists and matches the expected data type.
                Returns the loaded object or None if the file is missing or invalid.

            - save(data: Any, filename: str) -> None
                Saves a Python object (must be JSON-serializable) to the specified file path.
                Overwrites the file if it already exists.

        Usage Example
        -------------

        from useful_tools import UsefulTools

        # Save data
        data = [{"source": "cnn", "title": "News headline"}]
        UsefulTools.JsonCache.save(data, "cache_file.json")

        # Load data
        loaded_data = UsefulTools.JsonCache.load("cache_file.json", expected_type=list)
        if loaded_data is not None:
            print("Cache hit:", len(loaded_data), "items")
        else:
            print("Cache miss or file corrupted.")

    """


    class JsonCache:
        @staticmethod
        def load(filename: str, expected_type: type = list):
            """
            Load JSON data from file. Optionally check that it matches expected_type (e.g., list or dict).
            """
            if not os.path.exists(filename):
                return None
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, expected_type):
                    print(f"Loaded cache from '{filename}' ({len(data)} items).")
                    return data
                else:
                    print(f"Cache file '{filename}' is invalid: expected {expected_type.__name__}.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Failed to load cache file '{filename}': {e}")
            return None

        @staticmethod
        def save(data, filename: str):
            """
            Save any JSON-serializable data to a file.
            """
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"Saved data to cache file '{filename}'.")
            except IOError as e:
                print(f"Failed to save cache to '{filename}': {e}")
        import json

        @staticmethod
        def compare_json_files(file1: str, file2: str) -> bool:
            """
            Compares the contents of two JSON files.

            :param file1: Path to the first JSON file
            :param file2: Path to the second JSON file
            :return: True if both JSON files have the same data, False otherwise
            """
            try:
                with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
                    data1 = json.load(f1)
                    data2 = json.load(f2)
                are_equal = data1 == data2
                if are_equal:
                    print(f"✅ The contents of '{file1}' and '{file2}' are the same.")
                else:
                    print(f"❌ The contents of '{file1}' and '{file2}' differ.")
                return are_equal
            except Exception as e:
                print(f"Error comparing files: {e}")
                return False
